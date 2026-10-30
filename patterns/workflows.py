"""Cele cinci pattern-uri din „Building Effective Agents” (Anthropic), plus apelul simplu ca bază.

Toate primesc un `LLM` și un `Trace`, deci costul și latența fiecărui pattern se pot compara direct.

| pattern | când merită | cât costă |
|---|---|---|
| prompt chaining | pași clari, fiecare verificabil | n apeluri secvențiale |
| routing | intrări de tipuri diferite, cu costuri diferite | +1 apel ieftin de clasificare |
| parallelization: sectioning | subprobleme independente | n apeluri, latența celui mai lent |
| parallelization: voting | decizii la care o greșeală costă mult | n × același apel |
| orchestrator-workers | nu știi dinainte câte subprobleme sunt | plan + n workeri + sinteză |
| evaluator-optimizer | criterii clare de calitate, iar iterația ajută | 2 apeluri per rundă |
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from patterns.llm import LLM, Trace, parallel

Prompt = Callable[[str], str]


class GateFailed(Exception):
    def __init__(self, step: str, output: str):
        super().__init__(f"poarta de după pasul {step!r} a respins rezultatul")
        self.step, self.output = step, output


@dataclass
class Step:
    name: str
    model: str
    prompt: Prompt                                # primește ieșirea pasului anterior
    gate: Callable[[str], bool] | None = None     # verificare programatică între pași


def single(llm: LLM, text: str, *, model: str, prompt: Prompt, trace: Trace, purpose: str = "single") -> str:
    return llm.complete(prompt(text), model=model, purpose=purpose, trace=trace)


def chain(llm: LLM, text: str, steps: list[Step], trace: Trace) -> str:
    current = text
    for step in steps:
        current = llm.complete(step.prompt(current), model=step.model, purpose=step.name, trace=trace)
        if step.gate is not None and not step.gate(current):
            raise GateFailed(step.name, current)  # oprim devreme: pașii următori ar costa degeaba
    return current


def route(llm: LLM, text: str, *, classifier_model: str, classify: Prompt, parse: Callable[[str], str],
          routes: dict[str, tuple[str, Prompt]], default: str, trace: Trace) -> tuple[str, str]:
    label = parse(llm.complete(classify(text), model=classifier_model, purpose="classify", trace=trace))
    if label not in routes:
        label = default
    model, prompt = routes[label]
    return label, llm.complete(prompt(text), model=model, purpose=f"route:{label}", trace=trace)


def sectioning(llm: LLM, text: str, sections: dict[str, tuple[str, Prompt]], trace: Trace) -> dict[str, str]:
    names = list(sections)
    jobs = [lambda t, n=n: llm.complete(sections[n][1](text), model=sections[n][0], purpose=n, trace=t) for n in names]
    return dict(zip(names, parallel(trace, jobs)))


def vote(llm: LLM, text: str, *, n: int, model: str, prompt: Prompt, parse: Callable[[str], str],
         trace: Trace, purpose: str = "classify") -> tuple[str, float]:
    """Majoritate din n apeluri independente; întoarce și acordul (util ca semnal de încredere)."""
    answers = parallel(trace, [lambda t: parse(llm.complete(prompt(text), model=model, purpose=purpose, trace=t))
                               for _ in range(n)])
    label, count = Counter(answers).most_common(1)[0]
    return label, count / n


def orchestrator_workers(llm: LLM, task: str, *, orchestrator_model: str, plan: Prompt,
                         parse_plan: Callable[[str], list[str]], worker_model: str,
                         worker: Callable[[str, str], str], synth: Callable[[str, list[str]], str],
                         trace: Trace, max_workers: int = 6) -> tuple[list[str], str]:
    subtasks = parse_plan(llm.complete(plan(task), model=orchestrator_model, purpose="plan", trace=trace))[:max_workers]
    results = parallel(trace, [lambda t, s=s: llm.complete(worker(task, s), model=worker_model, purpose="worker", trace=t)
                               for s in subtasks])
    return subtasks, llm.complete(synth(task, results), model=orchestrator_model, purpose="synth", trace=trace)


def evaluator_optimizer(llm: LLM, task: str, *, generator_model: str, generate: Callable[[str, str], str],
                        evaluator_model: str, evaluate: Callable[[str, str], str],
                        parse_eval: Callable[[str], tuple[bool, str]], trace: Trace,
                        max_rounds: int = 3) -> tuple[str, int, bool]:
    feedback, draft = "", ""
    for round_ in range(1, max_rounds + 1):
        draft = llm.complete(generate(task, feedback), model=generator_model, purpose="generate", trace=trace)
        ok, feedback = parse_eval(llm.complete(evaluate(task, draft), model=evaluator_model, purpose="evaluate",
                                               trace=trace))
        if ok:
            return draft, round_, True
    return draft, max_rounds, False
