"""Lightweight HSM plumbing: event dispatch, future helpers, and I/O activity wrappers.

Only activities perform I/O and can fail (handled via `hsm/error` or typed outcome events).
Pure guards, effects, entries, and exits run directly without internal-fault self-healing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import hsm

from .types import EXITS, STATE_HEAD, Aborted, TransitionRefused

_T = hsm.Transition
_S = hsm.Target
_On = hsm.On
_G = hsm.Guard
_E = hsm.ErrorEvent
Effect = hsm.Effect
Entry = hsm.Entry
Exit = hsm.Exit
Guard = hsm.Guard


def described(error: BaseException) -> str:
    """Format an error as '<type>: <message>', guarding against broken __str__/__name__."""
    try:
        name = type(error).__name__
        name = name if isinstance(name, str) else "error"
    except EXITS:
        raise
    except BaseException:  # noqa: BLE001
        name = "error"
    try:
        text = str(error)
    except EXITS:
        raise
    except BaseException:  # noqa: BLE001
        return name
    return f"{name}: {text}" if text else name


def _resolve(reply: asyncio.Future | None, value: Any = None) -> None:
    if reply is not None and not reply.done():
        reply.set_result(value)


def _fail(reply: asyncio.Future | None, error: BaseException) -> None:
    if reply is not None and not reply.done():
        try:
            reply.set_exception(error)
        except TypeError:
            reply.set_exception(RuntimeError(described(error)))


def _send(target: hsm.Instance, event: hsm.Event) -> None:
    hsm.Dispatch(None, target, event)


def _forward(target: hsm.Instance | None, name: str, data: Any = None) -> None:
    if target is not None:
        hsm.Dispatch(None, target, hsm.Event(name=name, data=data))


def _self(inst: hsm.Instance, name: str, data: Any = None) -> None:
    hsm.Dispatch(None, inst, hsm.Event(name=name, data=data))


def _session(inst, name: str, data: Any = None) -> None:
    _forward(target=inst.machines.session, name=name, data=data)


def _note(inst, method: str, keep: BaseException | None = None, **args: Any) -> None:
    """Fire-and-forget write routed through the scribe machine."""
    _forward(target=inst.machines.scribe, name="note", data=((method, args), keep))


def _failure(ctx, error: BaseException) -> BaseException:
    """Classify an exception raised inside an activity.

    Interpreter exits propagate. A CancelledError means the state is being left only when `ctx` is
    done (hsm marks it before cancelling); any other cancellation (from the host or an awaited
    call) is uncancelled and returned as a RuntimeError so it is handled as an activity failure.
    """
    if isinstance(error, EXITS):
        raise error
    if isinstance(error, asyncio.CancelledError):
        if ctx.is_done():
            raise error
        task = asyncio.current_task()
        while task is not None and task.cancelling():
            task.uncancel()
        failure = RuntimeError(f"cancelled from outside: {described(error)}")
        failure.__cause__ = error
        return failure
    return error


def Activity(work: Callable, done: str = "") -> hsm.RedefinableBehaviors:
    """Wrap an async activity so non-Exception failures become `Aborted` (dispatched as `hsm/error`)."""
    async def activity(ctx, inst, event) -> None:
        try:
            await work(ctx=ctx, inst=inst, event=event)
        except BaseException as error:  # noqa: BLE001
            failure = _failure(ctx, error)
            if failure is error and isinstance(error, Exception):
                raise
            raise Aborted(described(failure)) from error
    activity.__name__ = work.__name__
    return hsm.Activity(activity)


def io(done: str, work: Callable) -> hsm.RedefinableBehaviors:
    """Wrap a synchronous I/O function as a state activity dispatching `done` with its return value."""
    async def activity(ctx, inst, event) -> None:
        try:
            result = work(inst=inst, event=event)
        except BaseException as error:  # noqa: BLE001
            failure = _failure(ctx, error)
            if failure is error and isinstance(error, Exception):
                raise
            raise Aborted(described(failure)) from error
        inst.dispatch(inst.context(), hsm.Event(name=done, data=result))
    activity.__name__ = f"io_{done}"
    return hsm.Activity(activity)


def mirrored(status: str) -> hsm.RedefinableBehaviors:
    def mirror(ctx, inst, event) -> None:
        _note(inst=inst, method="mirror", tl=inst.timeline_id, status=status)
    mirror.__name__ = f"mirror_{status}"
    return Entry(mirror)


def into(status: str) -> hsm.RedefinableBehaviors:
    def mirror(ctx, inst, event) -> None:
        inst.ended = status
        _note(inst=inst, method="mirror", tl=inst.timeline_id, status=status)
    mirror.__name__ = f"mirror_into_{status}"
    return Effect(mirror)


def _reply_of(data: Any) -> asyncio.Future | None:
    if isinstance(data, asyncio.Future):
        return data
    if isinstance(data, tuple):
        return next((x for x in reversed(data) if isinstance(x, asyncio.Future)), None)
    return getattr(data, "reply", None)


def refuse_request(ctx, inst, event) -> None:
    where = inst.state().split("/", STATE_HEAD)[-1] or "stopped"
    _fail(reply=_reply_of(event.data), error=TransitionRefused(
        f"{getattr(inst, 'timeline_id', 'session')}: '{event.name}' refused in state {where}"))
