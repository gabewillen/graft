"""Secret filter for anything sent to a provider in the replay.

DROP rules (an event that matches is not replayed): credential formats (provider API keys,
GitHub/Slack tokens, cloud keys, private keys, bearer tokens) and exact occurrences of the values
of secret-named keys (KEY/TOKEN/SECRET/PASSWORD/PRIVATE/CREDENTIAL) in known .env files. Values of
other .env keys (model names, base URLs) are not secrets and do not match.
REDACT rule: `secret_assignment` (`NAME=value` with a secret-looking NAME) keeps NAME and replaces
the value with [REDACTED] wherever text is sent; it never drops an event.
Functions report rule names only; values are held in memory and never printed.
"""

from __future__ import annotations

import re

PATTERNS = {
    "openai_key": re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{24,}=*"),
}
DROP_RULES = tuple(PATTERNS)
ASSIGNMENT = re.compile(
    r"(?im)(^\s*(?:export\s+)?[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY)[A-Z0-9_]*\s*=\s*['\"]?)"
    r"([^\s'\"]{12,})")
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE|CREDENTIAL", re.I)


def env_values(paths: list[str]) -> list[str]:
    """Values of secret-named KEY=value lines in the given .env files (length >= 12)."""
    vals = []
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                v = v.strip().strip("'\"")
                if len(v) >= 12 and SECRET_NAME.search(k):
                    vals.append(v)
        except OSError:
            continue
    return vals


def scan(texts: list[str], known_values: list[str] | None = None) -> list[str]:
    """Names of the DROP rules that match any text (empty = the event may be replayed)."""
    hits = set()
    for t in texts:
        if not t:
            continue
        for name, rx in PATTERNS.items():
            if rx.search(t):
                hits.add(name)
        for v in known_values or []:
            if v in t:
                hits.add("env_value")
    return sorted(hits)


def redact_assignments(text: str) -> tuple[str, int]:
    """Replace the value of every secret-looking NAME=value with [REDACTED]; returns (text, count)."""
    return ASSIGNMENT.subn(lambda m: m.group(1) + "[REDACTED]", text)


def redact(text: str, known_values: list[str] | None = None) -> tuple[str, list[str]]:
    """Text with every rule's matches replaced (used on tool outputs), plus the rule names that fired."""
    text, n = redact_assignments(text)
    hits = ["secret_assignment"] * n
    for name, rx in PATTERNS.items():
        if rx.search(text):
            hits.append(name)
            text = rx.sub("[REDACTED]", text)
    for v in known_values or []:
        if v in text:
            hits.append("env_value")
            text = text.replace(v, "[REDACTED]")
    return text, hits
