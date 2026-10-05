"""Secret filter for anything sent to a provider in the replay.

Two checks: generic credential patterns (API keys, tokens, private keys, KEY=value assignments
with secret-looking names), and exact occurrences of values found in known .env files (passed in;
the values are only held in memory and never printed). `scan` returns the names of the matching
rules, never the matched text.
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
    "secret_assignment": re.compile(
        r"(?im)^\s*(?:export\s+)?[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD|PASSWD|PRIVATE_KEY)[A-Z0-9_]*\s*=\s*"
        r"['\"]?[^\s'\"]{12,}"),
}


def env_values(paths: list[str]) -> list[str]:
    """Values of KEY=value lines in the given .env files (length >= 12), for exact matching."""
    vals = []
    for p in paths:
        try:
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                v = line.split("=", 1)[1].strip().strip("'\"")
                if len(v) >= 12:
                    vals.append(v)
        except OSError:
            continue
    return vals


def scan(texts: list[str], known_values: list[str] | None = None) -> list[str]:
    """Names of the rules that match any text (empty = clean)."""
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


def redact(text: str, known_values: list[str] | None = None) -> tuple[str, list[str]]:
    """Text with secret matches replaced, plus the rule names that fired."""
    hits = []
    for name, rx in PATTERNS.items():
        if rx.search(text):
            hits.append(name)
            text = rx.sub("[REDACTED]", text)
    for v in known_values or []:
        if v in text:
            hits.append("env_value")
            text = text.replace(v, "[REDACTED]")
    return text, hits
