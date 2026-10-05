"""Replay pilot plumbing (experiments/replay): transcript loader, secret filter, read-only sandbox.
Fixtures are synthetic; no real transcripts."""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "experiments"))

from replay import loader, sandbox, secret_filter  # noqa: E402

FIX = os.path.join(HERE, "fixtures", "replay", "proj", "sess1", "subagents", "agent-afixture000001.jsonl")


def test_loader_finds_spawn_prefix_and_report():
    e = loader.load_event(FIX)
    assert e.spawn_tool_use_id == "toolu_spawn" and e.spawn_index == 3
    assert e.description == "Audit parser" and e.subagent_type == "Explore" and e.branch == "main"
    assert [t.role for t in e.parent_prefix] == ["user", "assistant"]
    assert e.parent_prefix[1].text == "I'll look at it."  # thinking, tool calls and results dropped
    assert e.final_report == "Report: parse_token lacks bounds checks."
    assert e.recorded_tools == {"Read": 1}
    assert any(r["kind"] == "tool_result" and "parse_token" in r["text"] for r in e.recorded_turns)


def test_loader_matches_by_prompt_without_meta(tmp_path):
    import shutil

    d = tmp_path / "proj"
    shutil.copytree(os.path.join(HERE, "fixtures", "replay", "proj"), d)
    os.remove(d / "sess1" / "subagents" / "agent-afixture000001.meta.json")
    e = loader.load_event(str(d / "sess1" / "subagents" / "agent-afixture000001.jsonl"))
    assert e.spawn_tool_use_id == "toolu_spawn"


def test_cap_prefix_keeps_recent_turns_from_a_user_turn():
    turns = [loader.Turn("user", "a" * 10), loader.Turn("assistant", "b" * 10), loader.Turn("user", "c" * 10),
             loader.Turn("assistant", "d" * 10)]
    assert [t.text[0] for t in loader.cap_prefix(turns, 25)] == ["c", "d"]


def test_secret_filter_drop_rules_and_env_values(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123\n# c\nSHORT_KEY=x\n"
                   "OPENAI_BASE_URL=https://api.example.com/v1\nBOT_MODEL=some-model-name-123\n")
    vals = secret_filter.env_values([str(env)])
    assert vals == ["abcdefghijklmnopqrstuvwxyz123"]  # only secret-named keys
    assert secret_filter.scan(["uses https://api.example.com/v1 and some-model-name-123"], vals) == []
    assert secret_filter.scan(["nothing to see"], vals) == []
    assert secret_filter.scan(["key sk-proj-" + "A" * 30], vals) == ["openai_key"]
    assert "env_value" in secret_filter.scan(["value abcdefghijklmnopqrstuvwxyz123 here"], vals)
    assert "private_key" in secret_filter.scan(["-----BEGIN RSA PRIVATE KEY-----"])
    red, hits = secret_filter.redact("x ghp_" + "a" * 36 + " y")
    assert "[REDACTED]" in red and hits == ["github_token"]


def test_secret_assignment_is_redacted_not_dropped():
    text = "export GITHUB_TOKEN=zzzzzzzzzzzzzzzzzz\nAPI_KEY='abcdefghijkl1234'\nkeep this line"
    assert secret_filter.scan([text]) == []  # never a drop rule
    red, n = secret_filter.redact_assignments(text)
    assert n == 2 and red == "export GITHUB_TOKEN=[REDACTED]\nAPI_KEY='[REDACTED]'\nkeep this line"
    red2, hits = secret_filter.redact(text)
    assert hits.count("secret_assignment") == 2 and "zzzz" not in red2


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "src").mkdir(parents=True)
    (r / "src" / "parser.cpp").write_text("int parse_token();\nint other();\n")
    (r / "secret.txt").write_text("token sk-proj-" + "B" * 30 + "\n")
    subprocess.run(["git", "init", "-q", str(r)], check=True)
    subprocess.run(["git", "-C", str(r), "add", "."], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], check=True)
    (tmp_path / "outside.txt").write_text("outside")
    return r


def test_sandbox_read_grep_glob_and_alias(repo):
    sb = sandbox.Sandbox(str(repo), alias_roots=["/orig/repo"])
    out, err = sb.execute("Read", {"file_path": "/orig/repo/src/parser.cpp", "offset": 2, "limit": 1})
    assert not err and out == "2\tint other();"
    out, err = sb.execute("Grep", {"pattern": "parse_token", "path": "src"})
    assert not err and "src/parser.cpp:1" in out
    out, err = sb.execute("Glob", {"pattern": "**/*.cpp"})
    assert out == "src/parser.cpp"
    out, err = sb.execute("Bash", {"command": "git log --oneline"})
    assert not err and "init" in out
    out, err = sb.execute("Bash", {"command": "cat /orig/repo/src/parser.cpp"})
    assert "parse_token" in out


@pytest.mark.parametrize("cmd", [
    "rm -rf src", "ls | head", "cat src/parser.cpp > x", "echo hi", "git push", "git checkout main",
    "git -c core.pager=less log", "find . -exec rm {} ;", "find . -delete", "rg --pre cat x", "ls $(pwd)",
    "cat ../outside.txt", "cat /etc/passwd", "git log --output=x", "curl https://example.com", "ls; rm x",
])
def test_sandbox_refuses_unsafe_commands(repo, cmd):
    sb = sandbox.Sandbox(str(repo))
    out, err = sb.execute("Bash", {"command": cmd})
    assert err and out.startswith("refused"), (cmd, out)
    assert os.path.exists(repo / "src" / "parser.cpp")


def test_sandbox_confines_read_and_redacts(repo):
    sb = sandbox.Sandbox(str(repo))
    out, err = sb.execute("Read", {"file_path": str(repo.parent / "outside.txt")})
    assert err and "outside the checkout" in out
    out, err = sb.execute("Read", {"file_path": "secret.txt"})
    assert "[REDACTED]" in out and "sk-proj" not in out and "openai_key" in sb.redactions


def test_structured_payload_kind():
    from replay.run import is_messages

    assert is_messages([{"role": "user", "content": "q"}])
    assert not is_messages([{"id": 0, "question": "q"}]) and not is_messages({"a": 1}) and not is_messages([])


# ------------------------------------------------------------------ writable sandbox + worktrees
from replay import worktree  # noqa: E402


def _commit(r, msg="c"):
    subprocess.run(["git", "-C", str(r), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(r), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg], check=True)


@pytest.fixture
def wt(repo, tmp_path):
    (repo / ".gitignore").write_text(".env\n")
    (repo / ".env").write_text("OPENAI_API_KEY=" + "q" * 30 + "\n")  # untracked, must never be copied
    (repo / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    _commit(repo)
    commit = worktree.git(str(repo), "rev-parse", "HEAD")
    w = worktree.DisposableWorktree(str(repo), commit, str(tmp_path / "wt"))
    path = w.__enter__()
    yield w, path
    w.remove()


def test_disposable_worktree_no_env_and_cleanup(repo, tmp_path):
    (repo / ".gitignore").write_text(".env\n")
    (repo / ".env").write_text("SECRET=1\n")
    _commit(repo)
    commit = worktree.git(str(repo), "rev-parse", "HEAD")
    path = str(tmp_path / "wt2")
    with worktree.DisposableWorktree(str(repo), commit, path) as p:
        assert os.path.isdir(p) and not os.path.exists(os.path.join(p, ".env"))
        assert p in worktree.git(str(repo), "worktree", "list")
    assert not os.path.exists(path) and path not in worktree.git(str(repo), "worktree", "list")
    try:
        with worktree.DisposableWorktree(str(repo), commit, path):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not os.path.exists(path)


def test_writable_sandbox_confinement(wt, tmp_path):
    w, path = wt
    sb = sandbox.Sandbox(path, writable=True)
    out, err = sb.execute("Edit", {"file_path": "src/parser.cpp", "old_string": "int other();", "new_string": "int other2();"})
    assert not err and "other2" in open(os.path.join(path, "src", "parser.cpp")).read()
    out, err = sb.execute("Write", {"file_path": "src/new.cpp", "content": "int n;"})
    assert not err and os.path.exists(os.path.join(path, "src", "new.cpp"))
    for bad in ({"file_path": str(tmp_path / "escape.txt"), "content": "x"}, {"file_path": "../escape.txt", "content": "x"},
                {"file_path": ".git/config", "content": "x"}):
        out, err = sb.execute("Write", bad)
        assert err and out.startswith("refused"), bad
    assert not (tmp_path / "escape.txt").exists()
    out, err = sb.execute("Edit", {"file_path": "src/parser.cpp", "old_string": "int", "new_string": "long"})
    assert err and "occurs" in out  # ambiguous edit refused
    diff = w.diff()
    assert worktree.diff_files(diff) == {"src/new.cpp", "src/parser.cpp"}
    ro = sandbox.Sandbox(path)
    out, err = ro.execute("Write", {"file_path": "x.txt", "content": "x"})
    assert err and "read-only" in out and "Edit" not in [t["name"] for t in ro.tools()]


def test_writable_allowlist_tests_and_status(wt):
    w, path = wt
    sb = sandbox.Sandbox(path, writable=True, test_commands=[[sys.executable, "-m", "pytest", "-q"]])
    out, err = sb.execute("Bash", {"command": "git status --short"})
    assert not err
    out, err = sb.execute("Bash", {"command": f"{sys.executable} -m pytest -q test_ok.py"})
    assert "(exit 0)" in out and sb.test_runs[-1]["returncode"] == 0
    for cmd in (f"{sys.executable} -c print(1)", "uv pip install requests", "git commit -m x", "git push",
                f"{sys.executable} -m pytest -q; rm -rf src"):
        out, err = sb.execute("Bash", {"command": cmd})
        assert err and out.startswith("refused"), cmd
    out, err = sandbox.Sandbox(path).execute("Bash", {"command": "git status"})
    assert err  # status only in writable mode


def test_resolve_checkout_superproject_and_submodule(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init", "-q", str(sub)], check=True)
    (sub / "a.py").write_text("x = 1\n")
    _commit(sub, "sub1")
    sup = tmp_path / "sup"
    sup.mkdir()
    subprocess.run(["git", "init", "-q", str(sup)], check=True)
    subprocess.run(["git", "-C", str(sup), "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub),
                    "lib"], check=True)
    _commit(sup, "add lib")
    ts = "2037-01-01T00:00:00Z"
    r = worktree.resolve_checkout(str(sup), f"{sup}/.claude/worktrees/w1/lib/pkg", None, ts)
    assert r["repo"] == os.path.join(str(sup), "lib") and r["subdir"] == "pkg" and r["commit"]
    assert r["how"].startswith("submodule lib: gitlink")
    r2 = worktree.resolve_checkout(str(sup), str(sup), "main", ts)
    assert r2["repo"] == str(sup) and r2["subdir"] == "" and r2["how"].startswith("superproject")
    # a superproject worktree populates its submodule as a nested worktree, and removes both
    path = str(tmp_path / "supwt")
    with worktree.DisposableWorktree(str(sup), r2["commit"], path) as p:
        assert os.path.exists(os.path.join(p, "lib", "a.py"))
    assert not os.path.exists(path)
    assert str(tmp_path / "supwt") not in worktree.git(os.path.join(str(sup), "lib"), "worktree", "list")


def test_diff_similarity():
    a = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n+foo\n-bar\n"
    b = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n+foo\n"
    s = worktree.diff_similarity(a, b)
    assert s["files_jaccard"] == 1.0 and 0 < s["line_similarity"] < 1


def test_run_helpers_reference_diff_redaction_and_summary():
    import json as _json

    from replay import run

    e = loader.load_event(FIX)
    e.recorded_turns += [
        {"role": "assistant", "kind": "tool_use", "name": "Write",
         "text": _json.dumps({"file_path": "/orig/repo/src/new.cpp", "content": "int a;\nint b;"})},
        {"role": "assistant", "kind": "tool_use", "name": "Edit",
         "text": _json.dumps({"file_path": "/orig/repo/src/parser.cpp", "old_string": "int parse_token();",
                              "new_string": "long parse_token();"})}]
    ref = run.reference_diff(e, "/orig/repo")
    assert worktree.diff_files(ref) == {"src/new.cpp", "src/parser.cpp"} and "+long parse_token();" in ref
    e.handoff += "\nexport GITHUB_TOKEN=zzzzzzzzzzzzzzzzzz"
    assert run.redact_event(e) == {"secret_assignment": 1} and "GITHUB_TOKEN=[REDACTED]" in e.handoff
    assert run.strip_subdir("diff --git a/pkg/x.py b/pkg/x.py", "pkg") == "diff --git a/x.py b/x.py"
    s = run.test_summary("(exit 1)\n..F\n1 failed, 2 passed in 0.1s")
    assert s == {"returncode": 1, "summary": "1 failed, 2 passed in 0.1s"}
    assert run.is_editing(e) is False


def test_recorded_test_paths():
    import json as _json

    from replay import run

    e = loader.load_event(FIX)
    for c in ("uv run pytest -q tests/devices -p no:cacheprovider 2>&1 | tail", "uv run pytest tests/bot/x.py::t -q",
              "uv run pytest -q tests/devices"):
        e.recorded_turns.append({"role": "assistant", "kind": "tool_use", "name": "Bash", "text": _json.dumps({"command": c})})
    assert run.recorded_test_paths(e) == ["tests/devices", "tests/bot/x.py"]


def test_edit_aliases_and_test_selection(tmp_path):
    from replay import run

    e = loader.load_event(FIX)
    e.agent_id, e.cwd = "abc", "/rec/wt/sub"
    al = run.edit_aliases(e, "/repo", {"repo": "/repo/sub", "subdir": ""}, "/repo/sub")
    assert "/rec/wt/sub/.claude/worktrees/agent-abc" in al and "/repo/.claude/worktrees/agent-abc/sub" in al
    (tmp_path / "tests" / "devices").mkdir(parents=True)
    assert run.select_test_paths(["tests", "tests/devices", "tests/gone"], [], str(tmp_path)) == (["tests/devices"], [])
    assert run.select_test_paths(["tests"], ["tests/devices/test_x.py"], str(tmp_path)) == (["tests/devices"], [])
    assert run.select_test_paths(["tests"], [], str(tmp_path)) == (["tests"], ["--ignore=tests/examples", "-x"])


def test_worktree_diff_includes_submodule_changes(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init", "-q", str(sub)], check=True)
    (sub / "a.py").write_text("x = 1\n")
    _commit(sub, "sub1")
    sup = tmp_path / "sup"
    sup.mkdir()
    subprocess.run(["git", "init", "-q", str(sup)], check=True)
    subprocess.run(["git", "-C", str(sup), "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(sub), "lib"],
                   check=True)
    _commit(sup, "add lib")
    commit = worktree.git(str(sup), "rev-parse", "HEAD")
    with worktree.DisposableWorktree(str(sup), commit, str(tmp_path / "w")) as p:
        open(os.path.join(p, "lib", "a.py"), "w").write("x = 2\n")
        w = None
    # re-enter to inspect the diff before removal
    w = worktree.DisposableWorktree(str(sup), commit, str(tmp_path / "w2"))
    p = w.__enter__()
    try:
        open(os.path.join(p, "lib", "a.py"), "w").write("x = 2\n")
        assert "lib/a.py" in worktree.diff_files(w.diff())
    finally:
        w.remove()
