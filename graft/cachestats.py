"""Prompt-cache accounting for the experiments: effective input, cost, and report tables.

Usage dicts come from `providers.UsageMixin` and hold disjoint buckets:
  uncached     prompt tokens billed at the full input price
  cached       prompt tokens read from the provider's prompt cache
  cache_write  prompt tokens written to the cache (Anthropic bills these at a premium)
  output       completion tokens (including reasoning)

"Effective input" is uncached + cached x READ_MULT + cache_write x WRITE_MULT, in full-price
input-token equivalents. The multipliers are ASSUMPTIONS (defaults 0.1 and 1.25, Anthropic's
5-minute-TTL ratios); real ratios differ by provider and model. Override them with
GRAFT_CACHED_READ_MULT and GRAFT_CACHE_WRITE_MULT.

Cost is printed only when prices are set (USD per 1M tokens): GRAFT_PRICE_IN_PER_M,
GRAFT_PRICE_CACHED_IN_PER_M, GRAFT_PRICE_CACHE_WRITE_PER_M, GRAFT_PRICE_OUT_PER_M.
Missing cached/write prices fall back to the input price times the multipliers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str) -> float | None:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else None


def _or(v: float | None, default: float) -> float:
    return default if v is None else v


# Documented minimum cacheable prefix: OpenAI ~1024 tokens; Anthropic 512 on Claude Sonnet 5.5
# and the other newest models (1024-4096 on older ones). Used only to label the report.
MIN_CACHEABLE = {"openai": 1024, "anthropic": 512}


def min_cacheable(provider_name: str) -> int:
    return MIN_CACHEABLE.get(provider_name, 1024)


@dataclass(frozen=True)
class Pricing:
    read_mult: float = 0.1
    write_mult: float = 1.25
    price_in: float | None = None
    price_cached: float | None = None
    price_write: float | None = None
    price_out: float | None = None
    out_mult: float = 4.0

    @classmethod
    def from_env(cls) -> "Pricing":
        return cls(
            read_mult=_or(_env_float("GRAFT_CACHED_READ_MULT"), 0.1),
            write_mult=_or(_env_float("GRAFT_CACHE_WRITE_MULT"), 1.25),
            price_in=_env_float("GRAFT_PRICE_IN_PER_M"),
            price_cached=_env_float("GRAFT_PRICE_CACHED_IN_PER_M"),
            price_write=_env_float("GRAFT_PRICE_CACHE_WRITE_PER_M"),
            price_out=_env_float("GRAFT_PRICE_OUT_PER_M"),
            out_mult=_or(_env_float("GRAFT_OUTPUT_MULT"), 4.0))

    @property
    def priced(self) -> bool:
        return self.price_in is not None and self.price_out is not None

    def effective_input(self, u: dict) -> float:
        return u.get("uncached", 0) + u.get("cached", 0) * self.read_mult + u.get("cache_write", 0) * self.write_mult

    def effective_total(self, u: dict) -> float:
        """Effective input plus output weighted by out_mult (ASSUMED output/input price ratio)."""
        return self.effective_input(u) + u.get("output", 0) * self.out_mult

    def cost(self, u: dict) -> float | None:
        if not self.priced:
            return None
        pc = self.price_cached if self.price_cached is not None else self.price_in * self.read_mult
        pw = self.price_write if self.price_write is not None else self.price_in * self.write_mult
        return (u.get("uncached", 0) * self.price_in + u.get("cached", 0) * pc
                + u.get("cache_write", 0) * pw + u.get("output", 0) * self.price_out) / 1e6

    def assumptions(self) -> str:
        s = (f"effective input = uncached + {self.read_mult:g} x cached + {self.write_mult:g} x cache writes; "
             f"effective total = effective input + {self.out_mult:g} x output (ASSUMED multipliers, in "
             f"full-price input-token units; set GRAFT_CACHED_READ_MULT / GRAFT_CACHE_WRITE_MULT / "
             f"GRAFT_OUTPUT_MULT)")
        if self.priced:
            s += (f"; prices $/1M in={self.price_in:g} cached={self.price_cached} write={self.price_write} "
                  f"out={self.price_out:g}")
        else:
            s += "; cost n/a (set GRAFT_PRICE_IN_PER_M and GRAFT_PRICE_OUT_PER_M)"
        return s


def hit_rate(u: dict) -> float:
    prompt = u.get("uncached", 0) + u.get("cached", 0) + u.get("cache_write", 0)
    return u.get("cached", 0) / prompt if prompt else 0.0


COLUMNS = ("calls", "uncached in", "cached in", "hit rate", "cache writes", "output", "effective in",
           "effective total", "cost")


def table(rows: list[tuple[str, dict]], pricing: Pricing, per: int | None = None) -> list[str]:
    """One line per (label, usage). With `per`, also show per-run averages of the token columns."""
    head = ["role/arm"] + list(COLUMNS)
    body = []
    for label, u in rows:
        cost = pricing.cost(u)
        body.append([label, str(u.get("calls", 0)), str(u.get("uncached", 0)), str(u.get("cached", 0)),
                     f"{hit_rate(u):.1%}", str(u.get("cache_write", 0)), str(u.get("output", 0)),
                     f"{pricing.effective_input(u):.0f}", f"{pricing.effective_total(u):.0f}",
                     f"${cost:.4f}" if cost is not None else "n/a"])
        if per:
            pu = {k: v / per for k, v in u.items() if isinstance(v, (int, float))}
            pc = pricing.cost(pu)
            body.append([f"  per run (/{per})", f"{pu.get('calls', 0):.1f}", f"{pu.get('uncached', 0):.0f}",
                         f"{pu.get('cached', 0):.0f}", "", f"{pu.get('cache_write', 0):.0f}",
                         f"{pu.get('output', 0):.0f}", f"{pricing.effective_input(pu):.0f}",
                         f"{pricing.effective_total(pu):.0f}", f"${pc:.5f}" if pc is not None else "n/a"])
    widths = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    fmt = lambda r: "  ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r))
    return [fmt(head)] + [fmt(r) for r in body]


def prefix_line(label: str, counts: dict | None, min_cacheable: int = 1024) -> str:
    if not counts:
        return f"  {label}: token count unavailable"
    est = "" if counts.get("exact") else " (estimate: chars/4)"
    tas = counts["tools_and_system"]
    verdict = "reaches" if tas >= min_cacheable else "BELOW"
    return (f"  {label}: system prompt {counts['system']} tokens; tools+system {tas} tokens{est} -> "
            f"{verdict} the ~{min_cacheable}-token minimum cacheable prefix on its own")
