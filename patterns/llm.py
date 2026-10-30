"""Interfața LLM + contabilitatea care face pattern-urile comparabile: apeluri, tokeni, cost, latență.

Latența se calculează pe un **ceas virtual**: apelurile secvențiale se adună, iar cele paralele
contează cât cel mai lent dintre ele. Așa diferența dintre „chain” și „parallel” se vede în cifre,
fără să așteptăm secunde reale în teste.
"""

from __future__ import annotations

import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}


@dataclass
class Call:
    model: str
    purpose: str
    input_tokens: int
    output_tokens: int
    latency_s: float

    @property
    def cost(self) -> float:
        p_in, p_out = PRICES[self.model]
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1_000_000


@dataclass
class Trace:
    calls: list[Call] = field(default_factory=list)
    latency_s: float = 0.0  # pe ceasul virtual

    @property
    def cost(self) -> float:
        return sum(c.cost for c in self.calls)

    @property
    def tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)


class LLM(Protocol):
    def complete(self, prompt: str, *, model: str, purpose: str, trace: Trace) -> str: ...


def parallel(trace: Trace, jobs: list[Callable[[Trace], Any]]) -> list[Any]:
    """Rulează joburile în paralel. Fiecare are subtrace-ul lui; latența adăugată e maximul lor."""
    subs = [Trace() for _ in jobs]
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        results = list(pool.map(lambda pair: pair[0](pair[1]), zip(jobs, subs)))
    for s in subs:
        trace.calls += s.calls
    trace.latency_s += max((s.latency_s for s in subs), default=0.0)
    return results


class SimLLM:
    """LLM simulat pentru benchmark și teste.

    `skills` mapează un scop (`purpose`) la o funcție (prompt, rng) -> text. Latența și tokenii
    sunt modelați realist: Haiku e rapid și ieftin, Opus e lent și scump. `error_rate` per model
    simulează greșeli independente între apeluri; pe asta se bazează câștigul votului.
    """

    LATENCY = {"claude-haiku-5-5": (0.4, 0.004), "claude-sonnet-5-5": (0.9, 0.012), "claude-opus-5-5": (1.6, 0.02)}
    # (prima latență în secunde, secunde per token de output)

    def __init__(self, skills: dict[str, Callable[[str, random.Random, bool], str]], *, seed: int = 0,
                 error_rate: dict[str, float] | None = None, context_tokens: int = 0) -> None:
        self.skills = skills
        self.context_tokens = context_tokens  # system prompt + politici trimise la fiecare apel
        self.seed = seed
        self.counts: dict[str, int] = {}
        self.lock = threading.Lock()
        self.error_rate = error_rate or {"claude-haiku-5-5": 0.2, "claude-sonnet-5-5": 0.1, "claude-opus-5-5": 0.05}

    def _rng(self, model: str, purpose: str, prompt: str) -> random.Random:
        # determinist și sigur la thread-uri: al k-lea apel cu același prompt dă mereu același rezultat
        key = f"{model}|{purpose}|{prompt}"
        with self.lock:
            k = self.counts.get(key, 0)
            self.counts[key] = k + 1
        digest = hashlib.sha256(f"{self.seed}|{key}|{k}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def complete(self, prompt: str, *, model: str, purpose: str, trace: Trace) -> str:
        rng = self._rng(model, purpose, prompt)
        wrong = rng.random() < self.error_rate[model]
        text = self.skills[purpose](prompt, rng, wrong)
        out_tokens = max(1, len(text) // 4)
        first, per_token = self.LATENCY[model]
        latency = first + per_token * out_tokens
        trace.calls.append(Call(model, purpose, self.context_tokens + max(1, len(prompt) // 4), out_tokens, latency))
        trace.latency_s += latency
        return text


class ClaudeLLM:  # pragma: no cover - cere cheie API
    def __init__(self, client: Any = None) -> None:
        import anthropic
        self.client = client or anthropic.Anthropic()

    def complete(self, prompt: str, *, model: str, purpose: str, trace: Trace) -> str:
        import time
        t0 = time.perf_counter()
        msg = self.client.messages.create(model=model, max_tokens=4096, output_config={"effort": "low"},
                                          messages=[{"role": "user", "content": prompt}])
        latency = time.perf_counter() - t0
        trace.calls.append(Call(model, purpose, msg.usage.input_tokens, msg.usage.output_tokens, latency))
        trace.latency_s += latency
        return "".join(b.text for b in msg.content if b.type == "text")
