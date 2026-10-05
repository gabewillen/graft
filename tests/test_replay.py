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


def test_secret_filter(tmp_path):
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=abcdefghijklmnopqrstuvwxyz123\n# c\nSHORT=x\n")
    vals = secret_filter.env_values([str(env)])
    assert vals == ["abcdefghijklmnopqrstuvwxyz123"]
    assert secret_filter.scan(["nothing to see"], vals) == []
    assert secret_filter.scan(["key sk-proj-" + "A" * 30], vals) == ["openai_key"]
    assert "env_value" in secret_filter.scan(["value abcdefghijklmnopqrstuvwxyz123 here"], vals)
    assert "secret_assignment" in secret_filter.scan(["export GITHUB_TOKEN=zzzzzzzzzzzzzzzzzz"])
    assert "private_key" in secret_filter.scan(["-----BEGIN RSA PRIVATE KEY-----"])
    red, hits = secret_filter.redact("x ghp_" + "a" * 36 + " y")
    assert "[REDACTED]" in red and hits == ["github_token"]


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
