"""Shape proof for what graft's host interface emits: every object is checked against the vendored
A2A v1.0.1 JSON Schema (tests/a2a/a2a-v1.0.1.json), and against what that schema leaves out: the
proto's REQUIRED fields, camelCase names only, and enum names only (tests/a2a/SOURCE.md)."""

import json
import os
import re

import jsonschema
from referencing import Registry, Resource

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = json.load(open(os.path.join(HERE, "a2a", "a2a-v1.0.1.json")))
DEFINITIONS = BUNDLE["definitions"]


def _ref_name(title: str) -> str:
    """The file-style $ref a definition is known by in the bundle (`lf.a2a.v1.TaskStatus.jsonschema.json`)."""
    name = title.replace(" ", "")
    package = "google.protobuf" if name in ("Struct", "Value", "Timestamp") else "lf.a2a.v1"
    return f"{package}.{name}.jsonschema.json"


REGISTRY = Registry().with_resources(
    (_ref_name(title), Resource.from_contents(schema)) for title, schema in DEFINITIONS.items())

# The proto's REQUIRED field behaviours (a2a.proto, v1.0.1), by object, for the objects graft emits.
REQUIRED = {
    "Task": ("id", "status"),
    "Task Status": ("state",),
    "Message": ("messageId", "role", "parts"),
    "Task Status Update Event": ("taskId", "contextId", "status"),
    "Agent Card": ("name", "description", "supportedInterfaces", "version", "capabilities", "defaultInputModes",
                   "defaultOutputModes", "skills"),
    "Agent Skill": ("id", "name", "description", "tags"),
    "Agent Interface": ("url", "protocolBinding", "protocolVersion"),
    "List Tasks Response": ("tasks", "nextPageToken", "pageSize", "totalSize"),
}
PRESENT = ("tasks", "nextPageToken", "pageSize", "totalSize")
NESTED = {  # field -> the definition its value is (lists: each item)
    "status": "Task Status", "message": "Message", "history": "Message", "task": "Task",
    "statusUpdate": "Task Status Update Event", "supportedInterfaces": "Agent Interface",
    "skills": "Agent Skill", "tasks": "Task",
}
STATES = set(DEFINITIONS["Task Status"]["properties"]["state"]["anyOf"][1]["enum"])
ROLES = {"ROLE_USER", "ROLE_AGENT"}
CAMEL = re.compile(r"^[a-z][a-zA-Z0-9]*$")
FREE = ("metadata", "data", "structuredContent")  # values that are arbitrary JSON (Struct / Value)


def _strict(obj, name: str, where: str) -> list[str]:
    problems = []
    if not isinstance(obj, dict):
        return [f"{where}: not an object"]
    for field in REQUIRED.get(name, ()):
        value = obj.get(field)
        present_only = field in PRESENT  # an empty value is a value here (A2A §3.1.4)
        if field not in obj if present_only else value in (None, "", []):
            problems.append(f"{where}: {field} is required")
    for key, value in obj.items():
        if not CAMEL.match(key):
            problems.append(f"{where}: {key!r} is not a camelCase field name")
        if key in FREE:
            continue
        if key == "state" and value not in STATES:
            problems.append(f"{where}: state {value!r} is not a task state name")
        if key == "role" and value not in ROLES:
            problems.append(f"{where}: role {value!r} is not a role name")
        target = NESTED.get(key)
        if target == "Message" and name == "Stream Response":
            target = "Message"
        if target is not None:
            for i, item in enumerate(value if isinstance(value, list) else [value]):
                problems += _strict(item, target, f"{where}.{key}[{i}]" if isinstance(value, list) else f"{where}.{key}")
    return problems


def problems(obj, name: str) -> list[str]:
    """What is wrong with `obj` as the A2A object `name` (a definition title, e.g. "Task")."""
    validator = jsonschema.Draft202012Validator(DEFINITIONS[name], registry=REGISTRY)
    found = [f"schema: {e.json_path}: {e.message}" for e in validator.iter_errors(obj)]
    if name == "Stream Response":
        kinds = [k for k in ("task", "message", "statusUpdate", "artifactUpdate") if k in obj]
        if len(kinds) != 1:
            found.append(f"a StreamResponse holds exactly one of task, message, statusUpdate, artifactUpdate: {kinds}")
    return found + _strict(obj, name, name)


def check(obj, name: str) -> None:
    found = problems(obj, name)
    assert not found, f"not a valid A2A {name}:\n" + "\n".join(found) + "\n" + json.dumps(obj)[:2000]
