"""Read-only code tools for replayed delegations, confined to one checkout.

Tools (Claude Code names): Read(file_path, offset?, limit?), Grep(pattern, path?, glob?),
Glob(pattern, path?), Bash(command) with a strict allowlist:
  ls, cat, rg, grep, find, wc, head, tail, and git log/show/diff/ls-files.
No shell: the command is split with shlex and run with subprocess (no pipes, redirection,
substitution or chaining; any shell metacharacter is refused). No writes, no network
(find -exec/-delete/-fprint and rg --pre are refused; git runs with no pager and no external
diff). Every path argument must resolve inside the root; paths under the recorded original
checkout (`alias_roots`) are mapped onto the root. Output is capped and passed through the
secret filter before it is returned to a model.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shlex
import subprocess

from . import secret_filter

ALLOWED = {"ls", "cat", "rg", "grep", "find", "wc", "head", "tail", "git"}
GIT_ALLOWED = {"log", "show", "diff", "ls-files"}
FORBIDDEN_FIND = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
FORBIDDEN_RG = {"--pre", "--pre-glob", "--search-zip", "-z"}
FORBIDDEN_GIT = {"--output", "-o", "--ext-diff", "--textconv", "--exec", "-c", "--git-dir", "--work-tree",
                 "--upload-pack", "-C"}
META = re.compile(r"[;&|<>`$\n\r]|\$\(")
MAX_OUT = 8000

TOOLS = [
    {"name": "Read", "description": "Read a file (optionally from a line offset, for a number of lines).",
     "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"},
                                                       "limit": {"type": "integer"}}, "required": ["file_path"]}},
    {"name": "Grep", "description": "Search file contents with a regular expression (ripgrep).",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"},
                                                       "glob": {"type": "string"}}, "required": ["pattern"]}},
    {"name": "Glob", "description": "List files matching a glob pattern such as src/**/*.hpp.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                      "required": ["pattern"]}},
    {"name": "Bash", "description": "Run one read-only command (no pipes or redirection). Allowed: ls, cat, rg, grep, "
                                    "find, wc, head, tail, git log/show/diff/ls-files.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]


class SandboxError(ValueError):
    pass


class Sandbox:
    def __init__(self, root: str, alias_roots: list[str] | None = None, known_secrets: list[str] | None = None,
                 timeout: float = 20.0):
        self.root = os.path.realpath(root)
        self.aliases = sorted({os.path.realpath(a) for a in alias_roots or []} | set(alias_roots or []),
                              key=len, reverse=True)
        self.known = known_secrets or []
        self.timeout = timeout
        self.redactions: list[str] = []
        self.refusals = 0

    # ------------------------------------------------------------ paths
    def map_path(self, p: str) -> str:
        for a in self.aliases:
            if p == a or p.startswith(a + os.sep):
                p = self.root + p[len(a):]
                break
        full = os.path.realpath(p if os.path.isabs(p) else os.path.join(self.root, p))
        if full != self.root and not full.startswith(self.root + os.sep):
            raise SandboxError(f"path outside the checkout: {p}")
        return full

    def _map_arg(self, a: str) -> str:
        if a.startswith("-"):
            return a
        for al in self.aliases:
            if al in a:
                a = a.replace(al, self.root)
        if os.path.isabs(a):
            return self.map_path(a)
        if ".." in a.split(os.sep):
            self.map_path(a)  # raises if it escapes
        return a

    # ------------------------------------------------------------ tools
    def execute(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            out = getattr(self, f"_{name.lower()}")(args)
            err = False
        except SandboxError as e:
            self.refusals += 1
            out, err = f"refused: {e}", True
        except AttributeError:
            out, err = f"unknown tool {name}", True
        except (OSError, subprocess.SubprocessError) as e:
            out, err = f"error: {type(e).__name__}: {e}", True
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT] + f"\n... (truncated, {len(out)} chars)"
        out, hits = secret_filter.redact(out, self.known)
        self.redactions += hits
        return out, err

    def _read(self, a: dict) -> str:
        p = self.map_path(str(a.get("file_path", "")))
        if os.path.isdir(p):
            raise SandboxError("is a directory; use Glob or Bash ls")
        lines = open(p, errors="replace").read().splitlines()
        off = max(0, int(a.get("offset") or 1) - 1)
        lim = int(a.get("limit") or 2000)
        return "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines[off:off + lim], start=off))

    def _grep(self, a: dict) -> str:
        cmd = ["rg", "-n", "--no-heading", "--color", "never", "-e", str(a.get("pattern", ""))]
        if a.get("glob"):
            cmd += ["-g", str(a["glob"])]
        cmd.append(self._map_arg(str(a.get("path") or ".")))
        return self._run(cmd)

    def _glob(self, a: dict) -> str:
        base = self.map_path(str(a.get("path") or "."))
        pat = str(a.get("pattern", ""))
        out = []
        for dp, dns, fns in os.walk(base):
            dns[:] = [d for d in dns if d != ".git"]
            for f in fns:
                rel = os.path.relpath(os.path.join(dp, f), self.root)
                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(f, pat) or fnmatch.fnmatch(rel, pat.replace("**/", "")):
                    out.append(rel)
            if len(out) > 2000:
                break
        return "\n".join(sorted(out)) or "(no matches)"

    def _bash(self, a: dict) -> str:
        return self._run(self.check_command(str(a.get("command", ""))))

    def check_command(self, command: str) -> list[str]:
        if META.search(command):
            raise SandboxError("pipes, redirection, chaining and substitution are not allowed")
        try:
            argv = shlex.split(command)
        except ValueError as e:
            raise SandboxError(f"unparseable command: {e}") from e
        if not argv:
            raise SandboxError("empty command")
        if argv[0] == "cd" and len(argv) > 2 and argv[2] == "&&":
            raise SandboxError("chaining is not allowed")
        prog = os.path.basename(argv[0])
        if prog not in ALLOWED:
            raise SandboxError(f"command not allowed: {prog} (allowed: {sorted(ALLOWED)})")
        rest = argv[1:]
        if prog == "git":
            sub = next((x for x in rest if not x.startswith("-")), None)
            if sub not in GIT_ALLOWED or any(x.split("=")[0] in FORBIDDEN_GIT for x in rest):
                raise SandboxError(f"git subcommand not allowed (allowed: {sorted(GIT_ALLOWED)})")
            argv = ["git", "--no-pager", "-c", "core.pager=cat", "-c", "diff.external="] + rest
            if sub in ("log", "show", "diff"):
                argv.insert(argv.index(sub) + 1, "--no-ext-diff")
        if prog == "find" and any(x in FORBIDDEN_FIND for x in rest):
            raise SandboxError("find actions that write or execute are not allowed")
        if prog == "rg" and any(x.split("=")[0] in FORBIDDEN_RG for x in rest):
            raise SandboxError("rg preprocessors are not allowed")
        return [argv[0]] + [self._map_arg(x) for x in argv[1:]]

    def _run(self, argv: list[str]) -> str:
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "GIT_PAGER": "cat", "PAGER": "cat",
               "GIT_CONFIG_NOSYSTEM": "1", "HOME": self.root, "GIT_TERMINAL_PROMPT": "0"}
        p = subprocess.run(argv, cwd=self.root, capture_output=True, text=True, timeout=self.timeout, env=env,
                           errors="replace", stdin=subprocess.DEVNULL)
        out = p.stdout + (("\n" + p.stderr) if p.stderr else "")
        return out.strip() or f"(no output, exit {p.returncode})"
