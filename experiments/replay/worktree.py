"""Disposable detached git worktrees for replay arms, and checkout resolution for recorded events.

`resolve_checkout(repo, cwd, branch, timestamp)` maps a recorded cwd (possibly inside a Claude Code
worktree `<repo>/.claude/worktrees/<name>/<subdir>`) to the git repository that owns `<subdir>`
(the superproject, or a submodule such as `mosfet.py`), and picks the commit:
  1. superproject: last commit on the recorded branch before the spawn, else on any ref;
  2. submodule: the gitlink the superproject recorded at that commit, if it predates the spawn,
     else the submodule's last commit on any ref before the spawn.
The returned `how` says which rule applied.

`DisposableWorktree` adds a detached worktree at a commit and always removes it (`git worktree
remove --force` plus `prune`) on exit. It never copies untracked files such as `.env`; it records
whether a tracked `.env` exists at that commit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess


def git(repo: str, *args: str, check: bool = False) -> str:
    p = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def split_recorded_cwd(repo: str, cwd: str | None) -> str:
    """The subdirectory of the repo the recorded cwd points at ('' for the repo root)."""
    if not cwd:
        return ""
    m = re.match(re.escape(repo.rstrip("/")) + r"(?:/\.claude/worktrees/[^/]+)?(?:/(.*))?$", cwd.rstrip("/"))
    return (m.group(1) or "") if m else ""


def submodule_path(repo: str, subdir: str) -> str | None:
    """The submodule path that contains subdir, if any (gitlink mode 160000)."""
    parts = [x for x in subdir.split("/") if x]
    for i in range(len(parts), 0, -1):
        cand = "/".join(parts[:i])
        line = git(repo, "ls-files", "-s", "--", cand)
        if line.startswith("160000 "):
            return cand
    return None


def commit_before(repo: str, ts: str, branch: str | None) -> tuple[str, str]:
    if branch and git(repo, "rev-parse", "--verify", "--quiet", branch):
        c = git(repo, "rev-list", "-1", f"--before={ts}", branch)
        if c:
            return c, f"branch {branch} before spawn"
    c = git(repo, "rev-list", "-1", f"--before={ts}", "--all")
    return c, f"branch {branch!r} not resolvable; last commit on any ref before spawn"


def resolve_checkout(repo: str, cwd: str | None, branch: str | None, ts: str) -> dict:
    """{repo, subdir_in_checkout, commit, how}; commit '' when nothing resolves."""
    subdir = split_recorded_cwd(repo, cwd)
    sup_commit, how = commit_before(repo, ts, branch)
    sub = submodule_path(repo, subdir) if subdir else None
    if not sub:
        return {"repo": repo, "subdir": subdir, "commit": sup_commit, "how": f"superproject: {how}"}
    sub_repo = os.path.join(repo, sub)
    inner = subdir[len(sub):].lstrip("/")
    gitlink = ""
    if sup_commit:
        ls = git(repo, "ls-tree", sup_commit, "--", sub)
        gitlink = ls.split()[2] if ls.startswith("160000") else ""
    if gitlink and git(sub_repo, "cat-file", "-t", gitlink) == "commit":
        t = git(sub_repo, "show", "-s", "--format=%cI", gitlink)
        return {"repo": sub_repo, "subdir": inner, "commit": gitlink,
                "how": f"submodule {sub}: gitlink recorded at superproject {sup_commit[:10]} ({how}); committed {t}"}
    c, how2 = commit_before(sub_repo, ts, None)
    return {"repo": sub_repo, "subdir": inner, "commit": c, "how": f"submodule {sub}: {how2}"}


class DisposableWorktree:
    """with DisposableWorktree(repo, commit, path) as wt: ... -> the worktree path; removed on exit."""

    def __init__(self, repo: str, commit: str, path: str):
        self.repo, self.commit, self.path = repo, commit, path
        self.tracked_env = False

    def __enter__(self) -> str:
        if os.path.exists(self.path):
            raise RuntimeError(f"worktree path already exists: {self.path}")
        git(self.repo, "worktree", "add", "--detach", self.path, self.commit, check=True)
        self.tracked_env = bool(git(self.repo, "ls-tree", "-r", "--name-only", self.commit, "--", ".env"))
        return self.path

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        git(self.repo, "worktree", "remove", "--force", self.path)
        if os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=True)
        git(self.repo, "worktree", "prune")

    def diff(self) -> str:
        """Unified diff of the worktree against its commit, including new untracked files."""
        git(self.path, "add", "-N", ".")
        return git(self.path, "diff", "--no-ext-diff", self.commit)


def diff_files(diff: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", diff, re.M))


def diff_similarity(a: str, b: str) -> dict:
    """Files-touched overlap (Jaccard) and a line-level similarity of the added/removed lines."""
    import difflib

    fa, fb = diff_files(a), diff_files(b)
    jac = len(fa & fb) / len(fa | fb) if fa | fb else 1.0

    def changed(d: str) -> list[str]:
        return [ln for ln in d.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]

    la, lb = changed(a), changed(b)
    ratio = difflib.SequenceMatcher(None, la, lb, autojunk=False).ratio() if la or lb else 1.0
    return {"files_jaccard": jac, "line_similarity": ratio, "files_a": sorted(fa), "files_b": sorted(fb)}
