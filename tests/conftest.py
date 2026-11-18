"""Test-wide settings."""

import itertools

import pytest

import graft.harness as harness_mod


@pytest.fixture(autouse=True)
def counted_branch_ids(monkeypatch):
    """Branch ids are drawn from a seeded source in tests (spec: Clone, Ids): each harness names
    the grafts it is not given an id for c1, c2, ... in the order it draws them, so a test whose
    grafts are all unnamed sees each graft's name equal to its timeline id. Tests of the draw
    itself, and of names that differ from timeline ids, pass their own source or ids."""
    def counted():
        count = itertools.count(1)
        return lambda: f"c{next(count)}"
    monkeypatch.setattr(harness_mod, "new_names", counted)



@pytest.fixture(autouse=True)
def quoting_fork_calls(monkeypatch):
    """A scripted model's fork entry that names no messages and gives no reason is pointed at the
    user's messages in its view, as quotes (`_util.quoting`): the test default for a branch that
    starts from the conversation's requests (spec: Clone, Messages). Tests of what a branch sees
    give their entries messages or a reason of their own, and those are left as they are."""
    from graft.providers import FakeProvider
    from _util import quoting
    init = FakeProvider.__init__
    monkeypatch.setattr(FakeProvider, "__init__", lambda self, policy: init(self, quoting(policy)))
