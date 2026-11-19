"""Workspace-confined coding tools, skills, and inspection helpers for Graft's coding agent.

Used by both `graft acp` (Toad / Zed ACP server) and `graft code` (terminal REPL), as well as
Python hosts building coding agents on top of `GraftAgent`.

Tools (confined to `root`; writes inside `.git/` refused):
  - Read(file_path, offset?, limit?)
  - Grep(pattern, path?, glob?, max_matches?)
  - Glob(pattern, path?)
  - GitRead(subcommand, args?)
  - Edit(file_path, old_string, new_string, replace_all?)
  - Write(file_path, content)
  - Bash(command, timeout?) — async subprocess in its own process group so `stop` / CancelTask
    kills the entire command tree cleanly.

Skills:
  - `explore` (read-only): Read, Grep, Glob, GitRead — parallel detached research branches
  - `code` (full): Read, Grep, Glob, GitRead, Edit, Write, Bash — focused edits + verification
  - `shell` (runner): Bash, GitRead — detached test/build runs while main stays responsive
  - `search` (shipped automatically by `install()`): FTS5 search over merged branches' dropped
    Read/Grep/Bash outputs
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import os
import re
import shlex
import shutil
import signal
from pathlib import Path
from typing import Any, Callable

from .db import Skill, Store
from .skills import BASE_SYSTEM, BUILTIN
from .tools import CallToolResult, Tool, ToolContext

MAX_READ_BYTES = 32_000
MAX_GREP_BYTES = 24_000
MAX_BASH_BYTES = 24_000
MAX_GLOB_FILES = 300
DEFAULT_BASH_TIMEOUT = 120.0
MAX_BASH_TIMEOUT = 600.0

SKIP_DIRS = {".git", ".graft", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
GIT_READ_SUBCOMMANDS = frozenset({"status", "diff", "log", "show", "branch", "ls-files", "rev-parse"})
FORBIDDEN_GIT_FLAGS = frozenset({
    "-c", "--config-env", "--exec-path", "--git-dir", "--work-tree",
    "--ext-diff", "--textconv", "--output", "-o", "--upload-pack",
})

S = {"type": "string"}
I = {"type": "integer"}
B = {"type": "boolean"}


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required or ())}


def load_env(cwd: str | Path | None = None) -> None:
    """Load `.env` files (without overwriting existing env vars) and map `GRAFT_*` provider keys."""
    candidates: list[Path] = []
    if cwd is not None:
        candidates.append(Path(cwd).resolve() / ".env")
    here = Path.cwd().resolve() / ".env"
    if here not in candidates:
        candidates.append(here)
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if key and key not in os.environ:
                os.environ[key] = val
    # Map GRAFT_* prefixed credentials onto standard SDK env vars when unset.
    for src, dst in (
        ("GRAFT_OPENAI_API_KEY", "OPENAI_API_KEY"),
        ("GRAFT_OPENAI_BASE_URL", "OPENAI_BASE_URL"),
        ("GRAFT_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        ("GRAFT_ANTHROPIC_BASE_URL", "ANTHROPIC_BASE_URL"),
    ):
        if os.environ.get(src) and not os.environ.get(dst):
            os.environ[dst] = os.environ[src]


def coding_persona(cwd: str | Path) -> str:
    """System persona for Graft's coding agent operating in `cwd`."""
    return (
        BASE_SYSTEM
        + f"\n\nYou are a software engineering assistant working in workspace `{Path(cwd).resolve()}`.\n"
        "- Use `explore` branches (read-only) to investigate files, search code, or check git history; "
        "fan out parallel `detached` `explore` branches when researching independent modules or questions.\n"
        "- Use a `code` branch (`current`) to make edits and verify changes with tests.\n"
        "- Use a `shell` branch (`detached`) to run long tests or builds while staying responsive.\n"
        "- Set `wake: false` on a `current` branch when it finishes the user's request directly; keep "
        "`wake: true` for multi-stage pipelines (for example, `explore` first and then `code`).\n"
        "- Use `search` (or grant `search` to a branch) to recall exact snippets or command outputs "
        "from earlier merged branches without re-reading large files."
    )


CODING_SKILLS: list[Skill] = [
    Skill(
        name="explore",
        description="Read-only codebase exploration: read files, grep patterns, list files, and inspect git status/diff/log.",
        instructions=(
            "You are in read-only exploration mode (`Read`, `Grep`, `Glob`, `GitRead`).\n"
            "- Locate relevant files with `Glob`/`Grep` before reading full files.\n"
            "- Read targeted line ranges with `Read(file_path, offset, limit)` rather than dumping huge files.\n"
            "- Report exact file paths, line numbers, symbol names, and concise verified findings.\n"
            "- Never invent file contents or line numbers you did not observe."
        ),
        tools=["Read", "Grep", "Glob", "GitRead"],
    ),
    Skill(
        name="code",
        description="Edit files, create files, inspect code, and run shell commands/tests to implement and verify changes.",
        instructions=(
            "You are in implementation mode (`Read`, `Grep`, `Glob`, `GitRead`, `Edit`, `Write`, `Bash`).\n"
            "- Read the target files before editing so your `Edit` `old_string` matches verbatim.\n"
            "- Keep edits minimal, idiomatic, and scoped to the request.\n"
            "- Run relevant tests or checks via `Bash` after making edits to verify your changes.\n"
            "- Report the exact files changed and verification results concisely."
        ),
        tools=["Read", "Grep", "Glob", "GitRead", "Edit", "Write", "Bash"],
    ),
    Skill(
        name="shell",
        description="Run shell commands, builds, linters, or test suites (ideal for detached background runs).",
        instructions=(
            "You are in shell execution mode (`Bash`, `GitRead`).\n"
            "- Run the requested build, test, or inspection command via `Bash`.\n"
            "- Report the exit code, pass/fail summary, and key error lines concisely."
        ),
        tools=["Bash", "GitRead"],
    ),
]


class WorkspaceError(ValueError):
    """Raised when a tool call violates workspace confinement or input rules."""


def _truncate(text: str, limit: int, *, tail: bool = False) -> str:
    if len(text) <= limit:
        return text
    if tail:
        return f"... (truncated, showing last {limit} of {len(text)} chars)\n" + text[-limit:]
    return text[:limit] + f"\n... (truncated at {limit} of {len(text)} chars)"


def _summarize_args(tool: str, args: dict[str, Any], root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Return a concise human-readable title fragment and ACP tool locations."""
    if tool in ("Read", "Edit", "Write"):
        fp = str(args.get("file_path", "")).strip()
        locs: list[dict[str, Any]] = []
        shown = fp or "?"
        if fp:
            p = (Path(fp) if os.path.isabs(fp) else (root / fp)).resolve()
            try:
                shown = str(p.relative_to(root))
            except ValueError:
                shown = fp
            loc: dict[str, Any] = {"path": str(p)}
            if tool == "Read" and isinstance(args.get("offset"), int) and args["offset"] > 0:
                loc["line"] = args["offset"]
            locs.append(loc)
        suffix = f":{args['offset']}" if tool == "Read" and args.get("offset") else ""
        return f"{tool} {shown}{suffix}", locs
    if tool == "Grep":
        pat = str(args.get("pattern", ""))[:60]
        where = str(args.get("path") or ".")
        return f"Grep /{pat}/ in {where}", []
    if tool == "Glob":
        pat = str(args.get("pattern", ""))[:60]
        return f"Glob {pat}", []
    if tool == "GitRead":
        sub = str(args.get("subcommand", ""))
        extra = str(args.get("args", "")).strip()[:50]
        return f"git {sub}" + (f" {extra}" if extra else ""), []
    if tool == "Bash":
        cmd = " ".join(str(args.get("command", "")).split())[:80]
        return f"Bash: {cmd}", []
    return tool, []


class CodingWorkspace:
    """Workspace-confined coding toolset wired to `ToolContext` and optional live UI callbacks."""

    def __init__(self, root: str | Path, *, on_event: Callable[[dict[str, Any]], Any] | None = None):
        self.root = Path(root).resolve()
        if not self.root.exists() or not self.root.is_dir():
            raise WorkspaceError(f"workspace directory does not exist: {self.root}")
        self.on_event = on_event

    # ------------------------------------------------------------ path safety
    def resolve_path(self, raw: str, *, writable: bool = False) -> Path:
        raw = raw.strip()
        if not raw:
            raise WorkspaceError("file_path must be non-empty")
        candidate = Path(raw) if os.path.isabs(raw) else (self.root / raw)
        full = candidate.resolve()
        try:
            rel = full.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"path escapes workspace: {raw}") from None
        if writable and (rel.parts and rel.parts[0] == ".git"):
            raise WorkspaceError("writing inside .git is not allowed")
        return full

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    async def _emit(self, event: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        res = self.on_event(event)
        if inspect.isawaitable(res):
            await res

    # ------------------------------------------------------------ tool implementations
    async def _read(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        path = self.resolve_path(str(args.get("file_path", "")))
        if not path.exists():
            return f"File not found: {self.rel(path)}", True, None
        if path.is_dir():
            return f"{self.rel(path)} is a directory; use Glob to list files.", True, None
        raw_offset = args.get("offset", 1)
        raw_limit = args.get("limit", 300)
        offset = max(1, int(raw_offset)) if isinstance(raw_offset, int) and not isinstance(raw_offset, bool) else 1
        limit = max(1, min(2000, int(raw_limit))) if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else 300
        text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        start = offset - 1
        chunk = lines[start:start + limit]
        numbered = "\n".join(f"{i}\t{line}" for i, line in enumerate(chunk, start=offset))
        header = f"# {self.rel(path)} (lines {offset}-{start + len(chunk)} of {total})\n"
        out = header + numbered
        if start + len(chunk) < total:
            out += f"\n... ({total - (start + len(chunk))} more lines; pass offset={start + len(chunk) + 1} to continue)"
        return _truncate(out, MAX_READ_BYTES), False, None

    async def _grep(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return "pattern must be non-empty", True, None
        target = self.resolve_path(str(args.get("path") or "."))
        glob_pat = str(args.get("glob") or "").strip()
        raw_max = args.get("max_matches", 100)
        max_matches = max(1, min(500, int(raw_max))) if isinstance(raw_max, int) and not isinstance(raw_max, bool) else 100

        rg = shutil.which("rg")
        if rg:
            cmd = [rg, "-n", "--no-heading", "--color", "never", "--max-count", "50", "-e", pattern]
            if glob_pat:
                cmd += ["-g", glob_pat]
            cmd.append(str(target))
            out, rc = await self._run_argv(cmd, timeout=30.0)
            if rc == 1 and not out.strip():
                return "(no matches)", False, None
            if rc not in (0, 1):
                return f"rg error (exit {rc}): {out}", True, None
            lines = out.splitlines()
            # Rewrite absolute root prefixes to workspace-relative paths for clarity
            prefix = str(self.root) + os.sep
            cleaned = [ln[len(prefix):] if ln.startswith(prefix) else ln for ln in lines[:max_matches]]
            res = "\n".join(cleaned) or "(no matches)"
            if len(lines) > max_matches:
                res += f"\n... ({len(lines) - max_matches} more matches omitted)"
            return _truncate(res, MAX_GREP_BYTES), False, None

        # Pure-Python fallback when ripgrep is not installed
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"invalid regex: {e}", True, None

        def _py_grep() -> str:
            hits: list[str] = []
            files = [target] if target.is_file() else []
            if target.is_dir():
                for dp, dns, fns in os.walk(target):
                    dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
                    for fn in sorted(fns):
                        p = Path(dp) / fn
                        rel = self.rel(p)
                        if glob_pat and not (fnmatch.fnmatch(rel, glob_pat) or fnmatch.fnmatch(fn, glob_pat)):
                            continue
                        files.append(p)
            for p in files:
                try:
                    for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if rx.search(line):
                            hits.append(f"{self.rel(p)}:{idx}:{line}")
                            if len(hits) >= max_matches:
                                return "\n".join(hits) + "\n... (match limit reached)"
                except OSError:
                    continue
            return "\n".join(hits) or "(no matches)"

        return _truncate(await asyncio.to_thread(_py_grep), MAX_GREP_BYTES), False, None

    async def _glob(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return "pattern must be non-empty", True, None
        base = self.resolve_path(str(args.get("path") or "."))
        if not base.exists():
            return f"Path not found: {self.rel(base)}", True, None

        def _walk() -> str:
            matches: list[str] = []
            short = pattern.replace("**/", "")
            for dp, dns, fns in os.walk(base):
                dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
                for fn in sorted(fns):
                    rel = self.rel(Path(dp) / fn)
                    if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fn, pattern) or fnmatch.fnmatch(rel, short):
                        matches.append(rel)
                        if len(matches) >= MAX_GLOB_FILES:
                            return "\n".join(matches) + f"\n... (capped at {MAX_GLOB_FILES} files)"
            return "\n".join(matches) or "(no matches)"

        return await asyncio.to_thread(_walk), False, None

    async def _git_read(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        sub = str(args.get("subcommand", "")).strip()
        if sub not in GIT_READ_SUBCOMMANDS:
            return f"unsupported git subcommand {sub!r} (allowed: {sorted(GIT_READ_SUBCOMMANDS)})", True, None
        extra_str = str(args.get("args") or "").strip()
        try:
            extra = shlex.split(extra_str) if extra_str else []
        except ValueError as e:
            return f"invalid args: {e}", True, None
        for tok in extra:
            flag = tok.split("=", 1)[0]
            if flag in FORBIDDEN_GIT_FLAGS:
                return f"flag {flag!r} is not allowed in GitRead", True, None
        cmd = ["git", "--no-pager", "-c", "core.pager=cat", "-c", "diff.external=", sub]
        if sub in ("diff", "show", "log"):
            cmd.append("--no-ext-diff")
        cmd += extra
        out, rc = await self._run_argv(cmd, timeout=30.0)
        text = _truncate(out.strip() or "(empty)", MAX_GREP_BYTES)
        return (f"(exit {rc})\n{text}" if rc else text), rc != 0, None

    async def _edit(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        path = self.resolve_path(str(args.get("file_path", "")), writable=True)
        if not path.is_file():
            return f"File not found: {self.rel(path)}", True, None
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        replace_all = bool(args.get("replace_all", False))
        if not old:
            return "old_string must be non-empty", True, None
        before = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace")
        count = before.count(old)
        if count == 0:
            return f"old_string not found in {self.rel(path)}", True, None
        if count > 1 and not replace_all:
            return (f"old_string occurs {count} times in {self.rel(path)}; "
                    "pass a longer unique snippet or replace_all=true"), True, None
        after = before.replace(old, new) if replace_all else before.replace(old, new, 1)
        await asyncio.to_thread(path.write_text, after, encoding="utf-8")
        replaced = count if replace_all else 1
        diff = {"path": str(path), "oldText": before, "newText": after}
        return f"Edited {self.rel(path)} ({replaced} replacement{'s' if replaced != 1 else ''}).", False, diff

    async def _write(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        path = self.resolve_path(str(args.get("file_path", "")), writable=True)
        content = str(args.get("content", ""))
        before = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="replace") if path.is_file() else None
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        diff = {"path": str(path), "oldText": before, "newText": content}
        action = "Updated" if before is not None else "Created"
        return f"{action} {self.rel(path)} ({len(content)} bytes).", False, diff

    async def _bash(self, args: dict[str, Any]) -> tuple[str, bool, dict[str, Any] | None]:
        command = str(args.get("command", "")).strip()
        if not command:
            return "command must be non-empty", True, None
        raw_timeout = args.get("timeout", DEFAULT_BASH_TIMEOUT)
        timeout = (float(raw_timeout) if isinstance(raw_timeout, (int, float)) and not isinstance(raw_timeout, bool)
                   else DEFAULT_BASH_TIMEOUT)
        timeout = max(1.0, min(MAX_BASH_TIMEOUT, timeout))

        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(self.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await self._kill_group(proc)
            return f"Command timed out after {timeout:g}s.", True, None
        except asyncio.CancelledError:
            await self._kill_group(proc)
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        text = stdout_b.decode("utf-8", errors="replace").strip()
        out = f"(exit {rc})\n" + (_truncate(text, MAX_BASH_BYTES, tail=True) if text else "(no output)")
        return out, rc != 0, None

    async def _run_argv(self, argv: list[str], *, timeout: float) -> tuple[str, int]:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(self.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            await self._kill_group(proc)
            return f"command timed out after {timeout:g}s", -1
        except asyncio.CancelledError:
            await self._kill_group(proc)
            raise
        return stdout_b.decode("utf-8", errors="replace"), (proc.returncode if proc.returncode is not None else -1)

    @staticmethod
    async def _kill_group(proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2.0)
            except TimeoutError:
                pass

    # ------------------------------------------------------------ Tool wrappers
    def _wrap(self, name: str, kind: str, handler) -> Callable[[dict[str, Any], ToolContext], Any]:
        async def fn(args: dict[str, Any], call: ToolContext) -> CallToolResult:
            label = call.branch_name or call.branch
            summary, locs = _summarize_args(name, args, self.root)
            title = f"[{label}] {summary}"
            await self._emit({
                "phase": "start",
                "tool_use_id": call.tool_use_id,
                "branch": call.branch,
                "branch_name": call.branch_name,
                "tool": name,
                "kind": kind,
                "title": title,
                "locations": locs,
                "args": dict(args),
            })
            try:
                out, is_err, diff = await handler(args)
            except WorkspaceError as e:
                out, is_err, diff = f"Refused: {e}", True, None
            except asyncio.CancelledError:
                await self._emit({
                    "phase": "done",
                    "tool_use_id": call.tool_use_id,
                    "branch": call.branch,
                    "branch_name": call.branch_name,
                    "tool": name,
                    "kind": kind,
                    "title": title,
                    "locations": locs,
                    "output": "Stopped while running.",
                    "is_error": True,
                    "diff": None,
                })
                raise
            await self._emit({
                "phase": "done",
                "tool_use_id": call.tool_use_id,
                "branch": call.branch,
                "branch_name": call.branch_name,
                "tool": name,
                "kind": kind,
                "title": title,
                "locations": locs,
                "output": out,
                "is_error": is_err,
                "diff": diff,
            })
            return CallToolResult(content=[{"type": "text", "text": out}], is_error=is_err)
        return fn

    def tools(self) -> list[Tool]:
        return [
            Tool(
                name="Read",
                description="Read a file inside the workspace (numbered lines; supports optional 1-based offset and limit).",
                input_schema=_obj({"file_path": S, "offset": I, "limit": I}, ["file_path"]),
                fn=self._wrap("Read", "read", self._read),
            ),
            Tool(
                name="Grep",
                description="Search file contents in the workspace with a regex (ripgrep).",
                input_schema=_obj({"pattern": S, "path": S, "glob": S, "max_matches": I}, ["pattern"]),
                fn=self._wrap("Grep", "search", self._grep),
            ),
            Tool(
                name="Glob",
                description="List files in the workspace matching a glob pattern (e.g. '**/*.py').",
                input_schema=_obj({"pattern": S, "path": S}, ["pattern"]),
                fn=self._wrap("Glob", "search", self._glob),
            ),
            Tool(
                name="GitRead",
                description="Run a read-only git command (status, diff, log, show, branch, ls-files, rev-parse).",
                input_schema=_obj({"subcommand": S, "args": S}, ["subcommand"]),
                fn=self._wrap("GitRead", "read", self._git_read),
            ),
            Tool(
                name="Edit",
                description="Replace an exact string in an existing workspace file (old_string must be unique unless replace_all=true).",
                input_schema=_obj({"file_path": S, "old_string": S, "new_string": S, "replace_all": B},
                                  ["file_path", "old_string", "new_string"]),
                fn=self._wrap("Edit", "edit", self._edit),
            ),
            Tool(
                name="Write",
                description="Create or overwrite a file inside the workspace.",
                input_schema=_obj({"file_path": S, "content": S}, ["file_path", "content"]),
                fn=self._wrap("Write", "edit", self._write),
            ),
            Tool(
                name="Bash",
                description="Execute a shell command in the workspace root (cancellable; returns exit code + output).",
                input_schema=_obj({"command": S, "timeout": I}, ["command"]),
                fn=self._wrap("Bash", "execute", self._bash),
            ),
        ]


# ------------------------------------------------------------------ inspection helpers
def format_branches(store: Store) -> str:
    """Markdown table summarizing main and all forked branches."""
    rows = store.con.execute(
        "SELECT timeline_id, source_id, name, status, join_reason FROM timelines ORDER BY created_at"
    ).fetchall()
    if not rows:
        return "No branches yet."
    lines = ["| ID | Name | Parent | Status | Skills | Turns |", "|---|---|---|---|---|---:|"]
    for r in rows:
        tid = r["timeline_id"]
        name = r["name"] or ("main" if r["source_id"] is None else tid)
        parent = r["source_id"] or "-"
        status = r["status"] + (f" ({r['join_reason']})" if r["join_reason"] else "")
        skills = ", ".join(s["name"] for s in store.skills_for(tid) if s["name"] not in BUILTIN) or "-"
        turns = store.con.execute("SELECT COUNT(*) FROM context WHERE timeline_id=?", (tid,)).fetchone()[0]
        lines.append(f"| `{tid}` | `{name}` | `{parent}` | {status} | {skills} | {turns} |")
    return "\n".join(lines)


def format_usage(provider: Any) -> str:
    """Markdown summary of calls, tokens, prompt cache hit rate, and peak single-call input."""
    usage = getattr(provider, "usage", None)
    if not isinstance(usage, dict):
        return f"Provider `{getattr(provider, 'name', '?')}:{getattr(provider, 'model', '?')}` does not report usage."
    calls_log = list(getattr(provider, "call_log", []))
    total_in = usage.get("input", 0)
    cached = usage.get("cached", 0)
    cache_w = usage.get("cache_write", 0)
    uncached = usage.get("uncached", 0)
    out = usage.get("output", 0)
    calls = usage.get("calls", len(calls_log))
    peak = max((c.get("input", 0) for c in calls_log), default=0)
    hit = (100.0 * cached / total_in) if total_in else 0.0
    model = f"{getattr(provider, 'name', '?')}:{getattr(provider, 'model', '?')}"
    lines = [
        f"**Model:** `{model}`",
        f"- **Calls:** {calls}",
        f"- **Input:** {total_in:,} (cached {cached:,} [{hit:.1f}%], cache_write {cache_w:,}, uncached {uncached:,})",
        f"- **Peak single-call input:** {peak:,}",
        f"- **Output:** {out:,}",
    ]
    by_tag = getattr(provider, "by_tag", None)
    if isinstance(by_tag, dict) and by_tag:
        lines.append("")
        lines.append("| Role | Calls | Input | Cached | Output |")
        lines.append("|---|---:|---:|---:|---:|")
        for tag, u in sorted(by_tag.items()):
            lines.append(f"| `{tag}` | {u.get('calls', 0)} | {u.get('input', 0):,} | {u.get('cached', 0):,} | {u.get('output', 0):,} |")
    return "\n".join(lines)


def format_search(store: Store, query: str, limit: int = 8) -> str:
    """Run FTS5 search over committed history and format matching snippets."""
    query = query.strip()
    if not query:
        return "Usage: `/search <query>`"
    hits = store.search(query, limit=limit)
    if not hits:
        return f"No committed history matched `{query}`."
    lines = [f"Found {len(hits)} match(es) for `{query}`:"]
    for h in hits:
        where = "main" if h["source_id"] is None else (h["branch_name"] or h["owner_timeline"])
        what = f"tool:{h['tool_name']}" if h["kind"] == "tool_result" and h["tool_name"] else h["role"]
        snip = " ".join(str(h["snippet"]).split())
        lines.append(f"- `[#{h['row_id']}]` (`{where}`, {what}): {snip}")
    return "\n".join(lines)
