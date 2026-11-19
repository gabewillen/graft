"""The scribe machine: serializes fire-and-forget writes (status mirrors, trace lines, reports)."""

from __future__ import annotations

import hsm

from .plumbing import Effect, _E, _G, _On, _S, _T, _forward, _self, io


class Scribe(hsm.Instance):
    def __init__(self, machines):
        super().__init__()
        self.host = machines.host
        self.machines = machines
        self.note: tuple = ((), None)


def _scribe_write(inst: Scribe, event):
    (method, args), _keep = inst.note
    return getattr(inst.host, method)(**args)


def _take_note(ctx, inst: Scribe, event) -> None:
    inst.note = event.data


def _is_report(ctx, inst: Scribe, event) -> bool:
    return inst.note[0][0] == "report"


def _is_flush(ctx, inst: Scribe, event) -> bool:
    return inst.note[0][0] == "flush"


def _told_after_write(ctx, inst: Scribe, event) -> bool:
    return inst.note[1] is not None and not _is_report(ctx, inst, event) and not _is_flush(ctx, inst, event)


def _report_line(ctx, inst: Scribe, event) -> None:
    line, fails = event.data
    if line is not None:
        _forward(target=inst.machines.session, name="report", data=(line, tuple(fails)))


def _flushed(ctx, inst: Scribe, event) -> None:
    _forward(target=inst.machines.session, name="scribe_flushed")


def _tell_kept(ctx, inst: Scribe, event) -> None:
    error = inst.note[1]
    _self(inst=inst, name="note", data=(("report", dict(
        kind="machine", src=None, branches=[], error=error, quiet=False, background=False)), None))


def _report_write(ctx, inst: Scribe, event) -> None:
    line = dict(kind="mirror", src=None, branches=[], error=event.data, quiet=False, background=False)
    _self(inst=inst, name="note", data=(("report", line), None))


def _line_lost(ctx, inst: Scribe, event) -> None:
    _forward(target=inst.machines.session, name="failure", data=(event.data,))


SCRIBE = hsm.Define(
    "Scribe",
    hsm.Initial(_S("ready")),
    hsm.State("ready", _T(_On("note"), _S("../writing"), Effect(_take_note))),
    hsm.State(
        "writing", io(done="written", work=_scribe_write),
        hsm.Defer("note"),
        _T(_On("written"), _S("../ready"), _G(_is_report), Effect(_report_line)),
        _T(_On("written"), _S("../ready"), _G(_is_flush), Effect(_flushed)),
        _T(_On("written"), _S("../ready"), _G(_told_after_write), Effect(_tell_kept)),
        _T(_On("written"), _S("../ready")),
        _T(_On(_E), _S("../ready"), _G(_is_report), Effect(_line_lost)),
        _T(_On(_E), _S("../ready"), Effect(_report_write)),
    ),
)
