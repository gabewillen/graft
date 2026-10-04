"""API probe for spec Open Question 1.

Anthropic (skipped without ANTHROPIC_API_KEY):
  (a) Are historical tool_use/tool_result blocks accepted for tools NOT declared in
      the current request? (with another tool declared, and with no tools at all)
  (b) Are thinking blocks (with signatures) accepted when spliced into a different
      conversation? Control: replayed in their own conversation.
OpenAI (skipped without OPENAI_API_KEY):
  (a) Does Chat Completions accept historical assistant tool_calls + tool messages for
      tools not declared in the current request?

Usage: uv run python experiments/api_probe.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NOOP = {"name": "noop", "description": "Does nothing.", "input_schema": {"type": "object", "properties": {}}}
BINDING_BETA = "thinking-binding-controls-2026-08-01"


def outcome(fn) -> str:
    try:
        return "ACCEPTED: " + fn()
    except Exception as e:  # report the API's own words
        return f"REJECTED: {type(e).__name__}: {str(e)[:300]}"


def anthropic_probe() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[anthropic] SKIPPED: ANTHROPIC_API_KEY is not set")
        return
    import anthropic

    client = anthropic.Anthropic()
    model = os.environ.get("GRAFT_ANTHROPIC_MODEL", "claude-sonnet-5-5")
    hist = [
        {"role": "user", "content": "Write the homepage."},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_probe_1", "name": "site_write",
                                           "input": {"page": "home", "content": "Hello"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_probe_1", "content": "ok"},
                                     {"type": "text", "text": "In five words, what page did you write?"}]},
    ]

    def text_of(r):
        return " ".join(b.text for b in r.content if b.type == "text")[:120]

    print(f"[anthropic] model={model}")
    print("(a1) undeclared tool history, other tool declared ->",
          outcome(lambda: text_of(client.messages.create(model=model, max_tokens=300, messages=hist,
                                                         tools=[NOOP]))))
    print("(a2) undeclared tool history, no tools param    ->",
          outcome(lambda: text_of(client.messages.create(model=model, max_tokens=300, messages=hist))))

    # (b) produce thinking blocks in conversation A
    conv_a = [{"role": "user", "content": "Is 391 prime? Reason carefully, then answer in one line."}]
    try:
        ra = client.messages.create(model=model, max_tokens=4000, system="You are assistant A.",
                                    thinking={"type": "adaptive", "display": "summarized"},
                                    output_config={"effort": "high"}, messages=conv_a)
    except Exception as e:
        print(f"(b) could not produce thinking blocks: {type(e).__name__}: {str(e)[:300]}")
        return
    blocks = [b.to_dict() for b in ra.content]
    n_think = sum(b["type"] in ("thinking", "redacted_thinking") for b in blocks)
    print(f"(b) conversation A produced {n_think} thinking block(s)")
    if not n_think:
        print("(b) INCONCLUSIVE: no thinking block to splice")
        return
    follow = {"role": "user", "content": "Thanks. Now: is 391 odd?"}

    def call(system, messages):
        r = client.messages.create(model=model, max_tokens=2000, system=system,
                                   thinking={"type": "adaptive"}, messages=messages,
                                   extra_headers={"anthropic-beta": BINDING_BETA})
        extra = r.to_dict().get("input_transformations")
        return f"{text_of(r)!r} input_transformations={extra}"

    print("(b0) control: replay in its own conversation      ->",
          outcome(lambda: call("You are assistant A.", conv_a + [{"role": "assistant", "content": blocks},
                                                                 follow])))
    conv_b = [{"role": "user", "content": "Hi, I'm planning a dentist website."},
              {"role": "assistant", "content": "Happy to help."},
              {"role": "user", "content": "Quick question first: is 391 prime?"},
              {"role": "assistant", "content": blocks}, follow]
    print("(b1) spliced into another conversation             ->",
          outcome(lambda: call("You are assistant B, a website helper.", conv_b)))
    print("Interpretation: REJECTED (400) or a non-empty input_transformations means spliced thinking "
          "is not usable, so joins must exclude thinking rows (the harness's validator already does).")


def openai_probe() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("[openai] SKIPPED: OPENAI_API_KEY is not set")
        return
    import openai

    from graft.providers import pick_openai_model

    client = openai.OpenAI()
    model = pick_openai_model(client)
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write the homepage."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_probe_1", "type": "function",
            "function": {"name": "site_write", "arguments": '{"page": "home", "content": "Hello"}'}}]},
        {"role": "tool", "tool_call_id": "call_probe_1", "content": "ok"},
        {"role": "user", "content": "In five words, what page did you write?"},
    ]
    noop = {"type": "function", "function": {"name": "noop", "description": "Does nothing.",
                                             "parameters": {"type": "object", "properties": {}}}}
    extra = {"reasoning_effort": "none"} if model.startswith(("gpt-5.", "gpt-6")) else {}

    def go(**kw):
        r = client.chat.completions.create(model=model, messages=msgs, max_completion_tokens=200, **kw)
        return repr((r.choices[0].message.content or "")[:120])

    items = [
        {"role": "user", "content": "Write the homepage."},
        {"type": "function_call", "call_id": "call_probe_1", "name": "site_write",
         "arguments": '{"page": "home", "content": "Hello"}'},
        {"type": "function_call_output", "call_id": "call_probe_1", "output": "ok"},
        {"role": "user", "content": "In five words, what page did you write?"},
    ]
    rnoop = {"type": "function", "name": "noop", "description": "Does nothing.",
             "parameters": {"type": "object", "properties": {}}}
    effort = os.environ.get("GRAFT_REASONING_EFFORT", "low")

    def rgo(**kw):
        r = client.responses.create(model=model, instructions="You are a helpful assistant.", input=items,
                                    max_output_tokens=400, store=False, reasoning={"effort": effort}, **kw)
        return repr((r.output_text or "")[:120])

    print(f"[openai] model={model}")
    print("chat (a1) undeclared tool history, other tool declared ->", outcome(lambda: go(tools=[noop], **extra)))
    print("chat (a2) undeclared tool history, no tools param    ->", outcome(lambda: go(**extra)))
    print(f"responses (a1) undeclared tool history, other tool declared, effort={effort} ->",
          outcome(lambda: rgo(tools=[rnoop])))
    print(f"responses (a2) undeclared tool history, no tools param, effort={effort}    ->", outcome(lambda: rgo()))


if __name__ == "__main__":
    anthropic_probe()
    openai_probe()
