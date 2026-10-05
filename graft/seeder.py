"""Seed actor for the one-shot clone (GRAFT_SEEDER=model|jev).

Jev request design (reduced + chunked; documented limits: 64k tokens per request, 32k for the
state plus the longest question): the state is only the new task (purpose) and the user's latest
request. Each candidate row is its own Noul whose instructions carry the row plus its immediate
neighbour rows (one before, one after) for local reference. Questions are packed into batches whose
estimated size (state + questions, characters / 4) stays under GRAFT_SEED_TOKEN_BUDGET (default
24000), and batches run in parallel. A question that alone would exceed the budget has its message
truncated deterministically (recorded in the decision).

With `model` (default) the original names the seed itself (seed_quotes / seed_row_ids). With
`jev`, a TypeSafe Jev decision model picks the seed: one `system_one` request per clone with one
Noul ("does a clone doing <purpose> need this message?") per candidate row. Candidates are the
text rows of the source's working view, excluding harness notes. Rows scoring at or above
GRAFT_SEED_THRESHOLD (default 0.4) are seeded in view order, and the user's latest message (the
trigger) is always seeded. The original's own seed fields are then ignored, except as the
fallback when Jev fails. The question and state format match experiments/seed_eval.py, which
replays this decision offline.

Jev pricing (docs.typesafe.ai/models, verified 2026-10-04): jev-1.13.0 input $0.042 / 1M tokens,
output free. TYPESAFE_API_KEY is read by the SDK from the environment.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

JEV_MODEL = "jev-1.13.0"
DEFAULT_BUDGET = 24000  # estimated tokens per request; documented hard limits are 32k / 64k
NEIGHBOR_CHARS = 300
MESSAGE_CHARS = 1500
# Chosen offline (experiments/seed_eval.py, 140 stored long_compaction events, reduced design):
# t=0.4 -> 140/140 events with full fact recall (single and chunked); t=0.5 -> 139 / 138; t=0.7 -> 136.
DEFAULT_THRESHOLD = 0.4
NOUL_TRUE = ("The clone needs this message: it carries facts, names, numbers, requirements or "
             "preferences the new task should use, or it is the request for the task.")
NOUL_FALSE = "The clone can do the new task well without this message."


def seeder_name() -> str:
    # Default: jev when a TypeSafe key is available, else the original seeds itself.
    default = "jev" if os.environ.get("TYPESAFE_API_KEY") else "model"
    name = os.environ.get("GRAFT_SEEDER", default)
    if name not in ("model", "jev"):
        raise ValueError(f"GRAFT_SEEDER must be model or jev, got {name!r}")
    return name


def seed_threshold() -> float:
    try:
        return float(os.environ.get("GRAFT_SEED_THRESHOLD", DEFAULT_THRESHOLD))
    except ValueError:
        return DEFAULT_THRESHOLD


def is_candidate(row: dict) -> bool:
    return row["kind"] == "text" and not row["content"].startswith("(harness note")


def seed_budget() -> int:
    try:
        return int(os.environ.get("GRAFT_SEED_TOKEN_BUDGET", DEFAULT_BUDGET))
    except ValueError:
        return DEFAULT_BUDGET


def est_tokens(obj) -> int:
    import json as _json

    return len(_json.dumps(obj, default=str)) // 4 + 1


def reduced_state(purpose: str, trigger_text: str) -> dict:
    return {"new_task": purpose, "latest_user_request": trigger_text[:4000]}


def _side(c: dict | None) -> dict | None:
    return None if c is None else {"row_id": c["row_id"], "role": c["role"], "text": c["text"][:NEIGHBOR_CHARS]}


def reduced_question(candidates: list[dict], i: int, max_message_chars: int = MESSAGE_CHARS) -> dict:
    """The Noul instructions for candidate i: the row plus one neighbour on each side."""
    c = candidates[i]
    return {"question": "Does a clone doing the new task need this message from the conversation?",
            "row_id": c["row_id"], "role": c["role"], "message": c["text"][:max_message_chars],
            "before": _side(candidates[i - 1] if i > 0 else None),
            "after": _side(candidates[i + 1] if i + 1 < len(candidates) else None)}


def reduced_batches(purpose: str, candidates: list[dict], trigger_text: str,
                    budget: int | None = None) -> tuple[dict, list[dict[str, dict]], list[int]]:
    """(state, batches of {question name: instructions}, truncated row ids). Every batch's estimated
    state + questions fits the budget; a question too large alone is truncated deterministically."""
    budget = budget or seed_budget()
    state = reduced_state(purpose, trigger_text)
    room = budget - est_tokens(state) - est_tokens({"true": NOUL_TRUE, "false": NOUL_FALSE}) - 50
    batches: list[dict[str, dict]] = [{}]
    used = 0
    truncated: list[int] = []
    for i, c in enumerate(candidates):
        q = reduced_question(candidates, i)
        size = est_tokens(q) + est_tokens({"true": NOUL_TRUE, "false": NOUL_FALSE})
        if size > room:  # deterministic truncation of this row's own message
            keep = max(100, MESSAGE_CHARS - (size - room) * 4 - 200)
            q = reduced_question(candidates, i, keep)
            size = est_tokens(q) + est_tokens({"true": NOUL_TRUE, "false": NOUL_FALSE})
            truncated.append(c["row_id"])
        if used + size > room and batches[-1]:
            batches.append({})
            used = 0
        batches[-1][f"row_{c['row_id']}"] = q
        used += size
    return state, [b for b in batches if b], truncated


def jev_request(purpose: str, candidates: list[dict], trigger_text: str) -> tuple[dict, dict]:
    """FULL variant (offline evaluation only): the whole conversation as state, one Noul per
    candidate keyed row_<id>. Does not scale to long originals; the harness uses reduced_batches."""
    import typesafe_sdk

    state = {"new_task": purpose, "latest_user_request": trigger_text,
             "conversation": [{"row_id": c["row_id"], "role": c["role"], "text": c["text"][:2000]}
                              for c in candidates]}
    questions = {f"row_{c['row_id']}": typesafe_sdk.Noul(
        instructions={"question": "Does a clone doing the new task need this message from the conversation?",
                      "row_id": c["row_id"], "message": c["text"][:600]},
        criteria={"true": NOUL_TRUE, "false": NOUL_FALSE}) for c in candidates}
    return state, questions


@dataclass
class SeedDecision:
    scores: dict[int, float] = field(default_factory=dict)
    latency: float = 0.0
    input_tokens: int | None = None
    model: str | None = None
    error: str | None = None
    requests: int = 0
    truncated: list[int] = field(default_factory=list)


class JevSeeder:
    """Scores candidate rows with Jev (reduced state, chunked under the budget, batches in
    parallel). `client_factory()` must return an async context manager with
    `system_one(state=, questions=, model=)` (the SDK's AsyncTypeSafeClient by default)."""

    name = "jev"

    def __init__(self, client_factory=None, model: str = JEV_MODEL, timeout: float = 20.0,
                 budget: int | None = None):
        self.model, self.timeout, self.budget = model, timeout, budget
        self.client_factory = client_factory or self._sdk_client

    def _sdk_client(self):
        import typesafe_sdk

        return typesafe_sdk.AsyncTypeSafeClient(model=self.model, timeout=self.timeout)

    async def score(self, purpose: str, candidates: list[dict], trigger_text: str) -> SeedDecision:
        import asyncio

        t0 = time.perf_counter()
        try:
            import typesafe_sdk

            state, batches, truncated = reduced_batches(purpose, candidates, trigger_text, self.budget)
            crit = {"true": NOUL_TRUE, "false": NOUL_FALSE}
            async with self.client_factory() as client:
                resps = await asyncio.gather(*(client.system_one(
                    state=state, questions={k: typesafe_sdk.Noul(instructions=q, criteria=crit) for k, q in b.items()},
                    model=self.model) for b in batches))
            scores: dict[int, float] = {}
            tin = 0
            for r in resps:
                scores.update({int(k.split("_", 1)[1]): float(v.noul) for k, v in r.nouls.items()})
                tin = None if tin is None or getattr(r.usage, "input_tokens", None) is None else \
                    tin + r.usage.input_tokens
            return SeedDecision(scores, time.perf_counter() - t0, tin, getattr(resps[0], "model", self.model),
                                None, len(batches), truncated)
        except Exception as e:  # any provider failure -> caller falls back
            return SeedDecision({}, time.perf_counter() - t0, None, self.model,
                                f"{type(e).__name__}: {str(e)[:300]}")


def make_seeder():
    return JevSeeder() if seeder_name() == "jev" else None
