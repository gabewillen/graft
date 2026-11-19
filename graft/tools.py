"""The host's tools, in the shape of an MCP tool definition (spec: Host Interface, Tools).

A `Tool` is a definition (`name`, optional `title` and `description`, `inputSchema`, optional
`outputSchema` and `annotations`, as MCP's `Tool`) plus the host's function, which is not part of
the definition. graft calls the function with the call's arguments object; it returns a `str`, a
`CallToolResult` (or its MCP JSON form), or an awaitable of either. What the model reads of a
result is its text blocks, or, with none, its `structuredContent` as JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# MCP tool names: 1-128 characters of letters, digits, _ - . (the model APIs refuse anything else).
TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
# graft's own tools: a host tool may not take one of these names.
RESERVED = frozenset({"load_skill", "fork", "merge", "stop"})


@dataclass
class CallToolResult:
    """MCP's CallToolResult: `content` blocks (graft reads text blocks), optional
    `structuredContent`, and `isError`."""
    content: list[dict] = field(default_factory=list)
    structured_content: Any = None
    is_error: bool = False

    def to_dict(self) -> dict:
        out: dict = {"content": list(self.content), "isError": self.is_error}
        if self.structured_content is not None:
            out["structuredContent"] = self.structured_content
        return out

    @classmethod
    def from_value(cls, value: Any) -> "CallToolResult | None":
        """A function's return value as a result: a str is one text block; a CallToolResult or its
        MCP JSON form is taken as it is; anything else is None (not a result)."""
        if isinstance(value, str):
            return cls(content=[{"type": "text", "text": str.__str__(value)}])
        if isinstance(value, CallToolResult):
            return value
        if isinstance(value, dict) and isinstance(value.get("content", []), list) and set(value) <= {
                "content", "structuredContent", "isError", "_meta"}:
            blocks = value.get("content", [])
            if all(isinstance(b, dict) for b in blocks):
                return cls(content=list(blocks), structured_content=value.get("structuredContent"),
                           is_error=bool(value.get("isError", False)))
        return None

    def text(self) -> str:
        """What the model reads: the text blocks, joined; with none, the structured content as JSON."""
        texts = [b["text"] for b in self.content if b.get("type") == "text" and isinstance(b.get("text"), str)]
        if texts:
            return "\n".join(texts)
        if self.structured_content is not None:
            try:
                return json.dumps(self.structured_content)
            except (TypeError, ValueError):
                return repr(self.structured_content)[:2000]
        return ""


@dataclass
class Tool:
    """One host tool: an MCP tool definition and the function graft calls with the arguments."""
    name: str
    input_schema: dict
    fn: Callable[[dict], Any | Awaitable[Any]]
    description: str = ""
    title: str | None = None
    output_schema: dict | None = None
    annotations: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not TOOL_NAME.fullmatch(self.name):
            raise ValueError(f"tool name {self.name!r} is not an MCP tool name")
        if self.name in RESERVED:
            raise ValueError(f"tool name {self.name!r} is graft's own")
        if not isinstance(self.input_schema, dict) or self.input_schema.get("type") != "object":
            raise ValueError(f"tool {self.name}: inputSchema must be a JSON Schema object (type: object)")
        if not callable(self.fn):
            raise ValueError(f"tool {self.name}: fn must be callable")

    def definition(self) -> dict:
        """The MCP tool definition (without the function)."""
        out: dict = {"name": self.name, "description": self.description, "inputSchema": self.input_schema}
        for key, value in (("title", self.title), ("outputSchema", self.output_schema),
                           ("annotations", self.annotations)):
            if value is not None:
                out[key] = value
        return out


def by_name(tools: list[Tool]) -> dict[str, Tool]:
    """The tools by name; two tools with one name are refused."""
    out: dict[str, Tool] = {}
    for t in tools:
        if not isinstance(t, Tool):
            raise TypeError(f"tools must be Tool instances, got {type(t).__name__}")
        if t.name in out:
            raise ValueError(f"two tools are named {t.name!r}")
        out[t.name] = t
    return out
