"""The A2A (Agent2Agent) objects graft speaks to its host, version 1.0.1 (spec: Host Interface).

Each class is one A2A object in its ProtoJSON form: `to_dict()` gives exactly the JSON the
protocol defines (camelCase fields, enum names, unset fields left out) and `from_dict()` takes it
back, validating it whole, since what a host sends is untrusted input. Nothing here is graft's
own vocabulary: branches, forks and merges never appear in these objects.

Errors are the A2A errors with their JSON-RPC codes (A2A §5.4, §9.5): `A2AError` carries the code,
the message and the A2A name; `InvalidParams` (-32602) is a request that is not a valid object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = "1.0"  # the A2A protocol version this interface exposes (A2A v1.0.1)

# Task states (A2A TaskState), by their ProtoJSON names.
SUBMITTED = "TASK_STATE_SUBMITTED"
WORKING = "TASK_STATE_WORKING"
COMPLETED = "TASK_STATE_COMPLETED"
FAILED = "TASK_STATE_FAILED"
CANCELED = "TASK_STATE_CANCELED"
INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
REJECTED = "TASK_STATE_REJECTED"
AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
STATES = (SUBMITTED, WORKING, COMPLETED, FAILED, CANCELED, INPUT_REQUIRED, REJECTED, AUTH_REQUIRED)
TERMINAL = frozenset({COMPLETED, FAILED, CANCELED, REJECTED})
INTERRUPTED = frozenset({INPUT_REQUIRED, AUTH_REQUIRED})

ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"
ROLES = (ROLE_USER, ROLE_AGENT)

# Bounds on what a host may send (spec: Host Interface, Surface): untrusted input never reaches the
# machines unbounded.
MAX_TEXT_CHARS = 100_000  # the text of one user message (all its parts)
MAX_PARTS = 64
MAX_ID_CHARS = 256  # a messageId, taskId, contextId or tenant
MAX_METADATA_CHARS = 16_000  # metadata, as JSON
MAX_PAGE_SIZE = 100
MAX_HISTORY = 10_000


# ------------------------------------------------------------------ errors
class A2AError(Exception):
    """An A2A error: `code` is its JSON-RPC code, `name` its A2A name (A2A §3.3.2, §5.4)."""
    code = -32603
    name = "InternalError"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def to_dict(self) -> dict:
        """The JSON-RPC error object (A2A §9.5)."""
        return {"code": self.code, "message": self.message,
                "data": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": self.name,
                          "domain": "a2a-protocol.org"}]}


class InvalidParams(A2AError):
    code, name = -32602, "InvalidParamsError"


class TaskNotFound(A2AError):
    code, name = -32001, "TaskNotFoundError"


class TaskNotCancelable(A2AError):
    code, name = -32002, "TaskNotCancelableError"


class PushNotificationNotSupported(A2AError):
    code, name = -32003, "PushNotificationNotSupportedError"


class UnsupportedOperation(A2AError):
    code, name = -32004, "UnsupportedOperationError"


class ContentTypeNotSupported(A2AError):
    code, name = -32005, "ContentTypeNotSupportedError"


# ------------------------------------------------------------------ validation
def _obj(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise InvalidParams(f"{where} must be an object")
    return value


def _str(d: dict, key: str, where: str, *, required: bool = False, limit: int = MAX_ID_CHARS) -> str | None:
    value = d.get(key)
    if value is None or (value == "" and not required):
        if required:
            raise InvalidParams(f"{where}.{key} is required")
        return None
    if not isinstance(value, str):
        raise InvalidParams(f"{where}.{key} must be a string")
    if required and not value:
        raise InvalidParams(f"{where}.{key} must not be empty")
    if len(value) > limit:
        raise InvalidParams(f"{where}.{key} is longer than {limit} characters")
    return value


def _int(d: dict, key: str, where: str, *, low: int, high: int) -> int | None:
    value = d.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.lstrip("-").isdigit() and len(value) <= 12:
        value = int(value)  # ProtoJSON may carry an int32 as a string
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidParams(f"{where}.{key} must be an integer")
    if not low <= value <= high:
        raise InvalidParams(f"{where}.{key} must be between {low} and {high}")
    return value


def _bool(d: dict, key: str, where: str) -> bool | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise InvalidParams(f"{where}.{key} must be a boolean")
    return value


def _metadata(d: dict, where: str) -> dict | None:
    value = d.get("metadata")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidParams(f"{where}.metadata must be an object")
    try:
        size = len(json.dumps(value))
    except (TypeError, ValueError, RecursionError) as error:
        raise InvalidParams(f"{where}.metadata is not JSON: {type(error).__name__}") from None
    if size > MAX_METADATA_CHARS:
        raise InvalidParams(f"{where}.metadata is larger than {MAX_METADATA_CHARS} characters")
    return value


def _strings(d: dict, key: str, where: str, *, limit: int = 32) -> list[str]:
    value = d.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > limit or not all(
            isinstance(x, str) and len(x) <= MAX_ID_CHARS for x in value):
        raise InvalidParams(f"{where}.{key} must be a list of at most {limit} short strings")
    return list(value)


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


# ------------------------------------------------------------------ core objects
@dataclass
class Part:
    """A2A Part. graft takes and gives text parts only."""
    text: str
    metadata: dict | None = None
    media_type: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"text": self.text, "metadata": self.metadata, "mediaType": self.media_type})

    @classmethod
    def from_dict(cls, value: Any, where: str = "part") -> "Part":
        d = _obj(value, where)
        for kind in ("raw", "url", "data"):
            if d.get(kind) is not None:
                raise ContentTypeNotSupported(f"{where}: only text parts are supported, not {kind}")
        text = d.get("text")
        if not isinstance(text, str):
            raise InvalidParams(f"{where}.text must be a string")
        media = _str(d, "mediaType", where)
        if media not in (None, "text/plain"):
            raise ContentTypeNotSupported(f"{where}: media type {media[:40]!r} is not supported; send text/plain")
        return cls(text=text, metadata=_metadata(d, where), media_type=media)


@dataclass
class Message:
    """A2A Message."""
    message_id: str
    role: str
    parts: list[Part]
    context_id: str | None = None
    task_id: str | None = None
    metadata: dict | None = None
    extensions: list[str] = field(default_factory=list)
    reference_task_ids: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts)

    def to_dict(self) -> dict:
        return _drop_none({"messageId": self.message_id, "contextId": self.context_id, "taskId": self.task_id,
                           "role": self.role, "parts": [p.to_dict() for p in self.parts],
                           "metadata": self.metadata, "extensions": self.extensions or None,
                           "referenceTaskIds": self.reference_task_ids or None})

    @classmethod
    def from_dict(cls, value: Any, where: str = "message") -> "Message":
        d = _obj(value, where)
        role = d.get("role")
        if role not in ROLES:
            raise InvalidParams(f"{where}.role must be one of {', '.join(ROLES)}")
        parts = d.get("parts")
        if not isinstance(parts, list) or not parts:
            raise InvalidParams(f"{where}.parts must be a non-empty list")
        if len(parts) > MAX_PARTS:
            raise InvalidParams(f"{where}.parts has more than {MAX_PARTS} parts")
        message = cls(message_id=_str(d, "messageId", where, required=True), role=role,
                      parts=[Part.from_dict(p, f"{where}.parts[{i}]") for i, p in enumerate(parts)],
                      context_id=_str(d, "contextId", where), task_id=_str(d, "taskId", where),
                      metadata=_metadata(d, where), extensions=_strings(d, "extensions", where),
                      reference_task_ids=_strings(d, "referenceTaskIds", where))
        if sum(len(p.text) for p in message.parts) > MAX_TEXT_CHARS:
            raise InvalidParams(f"{where}: text is longer than {MAX_TEXT_CHARS} characters")
        return message


@dataclass
class TaskStatus:
    state: str
    message: Message | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"state": self.state, "message": self.message.to_dict() if self.message else None,
                           "timestamp": self.timestamp})

    @classmethod
    def from_dict(cls, value: Any, where: str = "status") -> "TaskStatus":
        d = _obj(value, where)
        if d.get("state") not in STATES:
            raise InvalidParams(f"{where}.state must be a task state name")
        message = d.get("message")
        return cls(state=d["state"], message=Message.from_dict(message, f"{where}.message") if message else None,
                   timestamp=_str(d, "timestamp", where))


@dataclass
class Task:
    id: str
    context_id: str
    status: TaskStatus
    history: list[Message] = field(default_factory=list)
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return _drop_none({"id": self.id, "contextId": self.context_id, "status": self.status.to_dict(),
                           "history": [m.to_dict() for m in self.history] or None, "metadata": self.metadata})

    @classmethod
    def from_dict(cls, value: Any, where: str = "task") -> "Task":
        d = _obj(value, where)
        history = d.get("history") or []
        if not isinstance(history, list):
            raise InvalidParams(f"{where}.history must be a list")
        return cls(id=_str(d, "id", where, required=True), context_id=_str(d, "contextId", where, required=True),
                   status=TaskStatus.from_dict(d.get("status"), f"{where}.status"),
                   history=[Message.from_dict(m, f"{where}.history[{i}]") for i, m in enumerate(history)],
                   metadata=_metadata(d, where))


@dataclass
class TaskStatusUpdateEvent:
    task_id: str
    context_id: str
    status: TaskStatus
    metadata: dict | None = None

    def to_dict(self) -> dict:
        return _drop_none({"taskId": self.task_id, "contextId": self.context_id, "status": self.status.to_dict(),
                           "metadata": self.metadata})

    @classmethod
    def from_dict(cls, value: Any, where: str = "statusUpdate") -> "TaskStatusUpdateEvent":
        d = _obj(value, where)
        return cls(task_id=_str(d, "taskId", where, required=True), context_id=_str(d, "contextId", where, required=True),
                   status=TaskStatus.from_dict(d.get("status"), f"{where}.status"), metadata=_metadata(d, where))


@dataclass
class StreamResponse:
    """A2A StreamResponse: exactly one of `task`, `message`, `status_update`. Each row of graft's
    event log is one of these (spec: Delivery)."""
    task: Task | None = None
    message: Message | None = None
    status_update: TaskStatusUpdateEvent | None = None

    def to_dict(self) -> dict:
        if self.task is not None:
            return {"task": self.task.to_dict()}
        if self.message is not None:
            return {"message": self.message.to_dict()}
        return {"statusUpdate": self.status_update.to_dict()}

    @classmethod
    def from_dict(cls, value: Any, where: str = "event") -> "StreamResponse":
        d = _obj(value, where)
        kinds = [k for k in ("task", "message", "statusUpdate") if d.get(k) is not None]
        if len(kinds) != 1:
            raise InvalidParams(f"{where} must hold exactly one of task, message, statusUpdate")
        kind, = kinds
        if kind == "task":
            return cls(task=Task.from_dict(d["task"], f"{where}.task"))
        if kind == "message":
            return cls(message=Message.from_dict(d["message"], f"{where}.message"))
        return cls(status_update=TaskStatusUpdateEvent.from_dict(d["statusUpdate"], f"{where}.statusUpdate"))


# ------------------------------------------------------------------ requests and responses
@dataclass
class SendMessageConfiguration:
    accepted_output_modes: list[str] = field(default_factory=list)
    history_length: int | None = None
    return_immediately: bool = False

    def to_dict(self) -> dict:
        return _drop_none({"acceptedOutputModes": self.accepted_output_modes or None,
                           "historyLength": self.history_length,
                           "returnImmediately": self.return_immediately or None})

    @classmethod
    def from_dict(cls, value: Any, where: str = "configuration") -> "SendMessageConfiguration":
        d = _obj(value, where)
        if d.get("taskPushNotificationConfig") is not None:
            raise PushNotificationNotSupported("push notifications are not supported")
        return cls(accepted_output_modes=_strings(d, "acceptedOutputModes", where),
                   history_length=_int(d, "historyLength", where, low=0, high=MAX_HISTORY),
                   return_immediately=bool(_bool(d, "returnImmediately", where)))


@dataclass
class SendMessageRequest:
    message: Message
    configuration: SendMessageConfiguration = field(default_factory=SendMessageConfiguration)
    metadata: dict | None = None
    tenant: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"message": self.message.to_dict(), "configuration": self.configuration.to_dict() or None,
                           "metadata": self.metadata, "tenant": self.tenant})

    @classmethod
    def from_dict(cls, value: Any, where: str = "params") -> "SendMessageRequest":
        d = _obj(value, where)
        if d.get("message") is None:
            raise InvalidParams(f"{where}.message is required")
        message = Message.from_dict(d["message"], f"{where}.message")
        if message.role != ROLE_USER:
            raise InvalidParams(f"{where}.message.role must be {ROLE_USER}")
        configuration = d.get("configuration")
        return cls(message=message,
                   configuration=SendMessageConfiguration.from_dict(configuration, f"{where}.configuration")
                   if configuration is not None else SendMessageConfiguration(),
                   metadata=_metadata(d, where), tenant=_str(d, "tenant", where))


@dataclass
class SendMessageResponse:
    task: Task

    def to_dict(self) -> dict:
        return {"task": self.task.to_dict()}


@dataclass
class GetTaskRequest:
    id: str
    history_length: int | None = None
    tenant: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"id": self.id, "historyLength": self.history_length, "tenant": self.tenant})

    @classmethod
    def from_dict(cls, value: Any, where: str = "params") -> "GetTaskRequest":
        d = _obj(value, where)
        return cls(id=_str(d, "id", where, required=True),
                   history_length=_int(d, "historyLength", where, low=0, high=MAX_HISTORY), tenant=_str(d, "tenant", where))


@dataclass
class CancelTaskRequest:
    id: str
    metadata: dict | None = None
    tenant: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"id": self.id, "metadata": self.metadata, "tenant": self.tenant})

    @classmethod
    def from_dict(cls, value: Any, where: str = "params") -> "CancelTaskRequest":
        d = _obj(value, where)
        return cls(id=_str(d, "id", where, required=True), metadata=_metadata(d, where), tenant=_str(d, "tenant", where))


@dataclass
class SubscribeToTaskRequest:
    id: str
    tenant: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"id": self.id, "tenant": self.tenant})

    @classmethod
    def from_dict(cls, value: Any, where: str = "params") -> "SubscribeToTaskRequest":
        d = _obj(value, where)
        return cls(id=_str(d, "id", where, required=True), tenant=_str(d, "tenant", where))


@dataclass
class ListTasksRequest:
    context_id: str | None = None
    status: str | None = None
    page_size: int | None = None
    page_token: str | None = None
    history_length: int | None = None
    include_artifacts: bool = False
    tenant: str | None = None

    def to_dict(self) -> dict:
        return _drop_none({"contextId": self.context_id, "status": self.status, "pageSize": self.page_size,
                           "pageToken": self.page_token, "historyLength": self.history_length,
                           "includeArtifacts": self.include_artifacts or None, "tenant": self.tenant})

    @classmethod
    def from_dict(cls, value: Any, where: str = "params") -> "ListTasksRequest":
        d = _obj(value if value is not None else {}, where)
        status = d.get("status")
        if status is not None and status not in STATES:
            raise InvalidParams(f"{where}.status must be a task state name")
        if d.get("statusTimestampAfter") is not None:
            raise UnsupportedOperation("statusTimestampAfter is not supported: task statuses carry no timestamp")
        return cls(context_id=_str(d, "contextId", where), status=status,
                   page_size=_int(d, "pageSize", where, low=1, high=MAX_PAGE_SIZE),
                   page_token=_str(d, "pageToken", where),
                   history_length=_int(d, "historyLength", where, low=0, high=MAX_HISTORY),
                   include_artifacts=bool(_bool(d, "includeArtifacts", where)), tenant=_str(d, "tenant", where))


@dataclass
class ListTasksResponse:
    tasks: list[Task]
    next_page_token: str
    page_size: int
    total_size: int

    def to_dict(self) -> dict:
        return {"tasks": [t.to_dict() for t in self.tasks], "nextPageToken": self.next_page_token,
                "pageSize": self.page_size, "totalSize": self.total_size}


# ------------------------------------------------------------------ discovery
@dataclass
class AgentSkill:
    id: str
    name: str
    description: str
    tags: list[str]

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "description": self.description, "tags": list(self.tags)}


@dataclass
class AgentInterface:
    url: str
    protocol_binding: str
    protocol_version: str = PROTOCOL_VERSION

    def to_dict(self) -> dict:
        return {"url": self.url, "protocolBinding": self.protocol_binding, "protocolVersion": self.protocol_version}


@dataclass
class AgentCard:
    name: str
    description: str
    version: str
    supported_interfaces: list[AgentInterface]
    skills: list[AgentSkill]
    streaming: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description,
                "supportedInterfaces": [i.to_dict() for i in self.supported_interfaces],
                "version": self.version,
                "capabilities": {"streaming": self.streaming, "pushNotifications": False, "extendedAgentCard": False},
                "defaultInputModes": ["text/plain"], "defaultOutputModes": ["text/plain"],
                "skills": [s.to_dict() for s in self.skills]}
