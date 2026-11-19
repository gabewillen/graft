"""Host tools in MCP's shape (spec: Host Interface, Tools): validation, definitions, results."""

import asyncio

import pytest

from graft.harness import Ran
from graft.tools import CallToolResult, Tool, by_name

SCHEMA = {"type": "object", "properties": {"q": {"type": "string"}}}


def noop(args):
    return "ok"


@pytest.mark.parametrize("name", ["", "has space", "x" * 129, "fork", "merge", "stop", "search", 7])
def test_a_tool_name_that_is_not_mcp_or_is_grafts_own_is_refused(name):
    with pytest.raises(ValueError):
        Tool(name=name, input_schema=SCHEMA, fn=noop)


def test_the_input_schema_must_be_an_object_schema_and_fn_callable():
    with pytest.raises(ValueError):
        Tool(name="t", input_schema={"type": "string"}, fn=noop)
    with pytest.raises(ValueError):
        Tool(name="t", input_schema=SCHEMA, fn="not callable")


def test_the_definition_is_mcps_tool_with_only_the_fields_given():
    plain = Tool(name="lookup", input_schema=SCHEMA, fn=noop, description="Find.")
    assert plain.definition() == {"name": "lookup", "description": "Find.", "inputSchema": SCHEMA}
    full = Tool(name="lookup", input_schema=SCHEMA, fn=noop, title="Lookup", output_schema={"type": "object"},
                annotations={"readOnlyHint": True})
    assert full.definition()["title"] == "Lookup"
    assert full.definition()["outputSchema"] == {"type": "object"}
    assert full.definition()["annotations"] == {"readOnlyHint": True}


def test_by_name_refuses_duplicates_and_non_tools():
    a = Tool(name="a", input_schema=SCHEMA, fn=noop)
    assert by_name([a]) == {"a": a}
    with pytest.raises(ValueError):
        by_name([a, Tool(name="a", input_schema=SCHEMA, fn=noop)])
    with pytest.raises(TypeError):
        by_name([{"name": "a"}])


def test_results_from_values():
    assert CallToolResult.from_value("hi").text() == "hi"
    r = CallToolResult.from_value({"content": [{"type": "text", "text": "x"}], "isError": True})
    assert r.text() == "x" and r.is_error is True
    assert CallToolResult.from_value({"content": [], "isError": "yes"}).is_error is False  # only true is true
    assert CallToolResult.from_value({"content": [], "structuredContent": {"n": 1}}).text() == '{"n": 1}'
    assert CallToolResult.from_value(42) is None
    assert CallToolResult.from_value({"content": [], "other": 1}) is None
    deep: dict = {}
    cur = deep
    for _ in range(200_000):
        cur["d"] = {}
        cur = cur["d"]
    assert CallToolResult.from_value({"content": [], "structuredContent": deep}).text() == \
        "(structured content that is not JSON)"


class _Host:
    """Only what call_tool reads."""

    def __init__(self, tools):
        self.tools = by_name(tools)


def _call(fn, args=None):
    from graft.harness import Host
    host = _Host([Tool(name="t", input_schema=SCHEMA, fn=fn)])
    return asyncio.run(Host.call_tool(host, "tl", {"name": "t", "input": args or {}}))


def test_a_sync_and_an_async_tool_are_called_with_the_arguments():
    assert _call(lambda a: f"got {a['q']}", {"q": "x"}) == (Ran("got x"), False)

    async def later(a):
        await asyncio.sleep(0)
        return {"content": [{"type": "text", "text": "later"}]}
    assert _call(later) == (Ran("later"), False)


def test_a_tool_that_raises_or_returns_no_result_ran_with_an_error():
    def boom(a):
        raise RuntimeError("nope")
    ran, is_error = _call(boom)
    assert is_error and "raised RuntimeError" in ran and "may have acted" in ran
    ran, is_error = _call(lambda a: 3)
    assert is_error and "returned a int" in ran


def test_a_result_that_cannot_be_read_is_an_error_not_a_crash():
    class Bad(str):
        def __str__(self):
            raise RuntimeError("no")

    class BadResult(CallToolResult):
        def text(self):
            raise RecursionError("deep")
    ran, is_error = _call(lambda a: BadResult(content=[]))
    assert is_error and "could not be read" in ran
    assert _call(lambda a: Bad("fine")) == (Ran("fine"), False)


def test_a_tool_the_host_no_longer_gives_is_not_run():
    from graft.harness import Host
    host = _Host([])
    assert asyncio.run(Host.call_tool(host, "tl", {"name": "gone", "input": {}})) == \
        (Ran("tool 'gone' is not available"), True)
