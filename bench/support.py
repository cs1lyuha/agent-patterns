"""Benchmark: 300 de tichete de suport, 7 abordări, aceleași metrici.

    python -m bench.support

SIMULARE, nu model real. `SimLLM` „știe” răspunsul corect (dintr-un tag ascuns în tichet)
și greșește independent, cu rata setată per model: Haiku 20%, Sonnet 10%, Opus 5%.
Tabelul arată deci **consecințele structurale** ale fiecărui pattern (câte apeluri, ce
latență, cum se compun erorile), nu calitatea unui model anume. Pe Claude, același cod
rulează cu `ClaudeLLM`.
"""

from __future__ import annotations

import random
import re
import statistics
from dataclasses import dataclass

from patterns.llm import SimLLM, Trace
from patterns.workflows import GateFailed, Step, chain, evaluator_optimizer, orchestrator_workers, route, single, vote

H, S, O = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"
FACTS = {
    "livrare": ["50 de lei", "1-3 zile"], "retur": ["30 de zile", "verificare"],
    "garantie": ["24 de luni", "14 zile"], "plata": ["rate 0%", "20 000 lei"], "tehnic": ["resetare", "service"],
}
CATS = list(FACTS)
COMPLEX = {"garantie", "tehnic"}
TEXTS = {
    "livrare": "Când ajunge coletul și cât costă transportul?", "retur": "Vreau să dau înapoi produsul, cum fac?",
    "garantie": "Laptopul s-a stricat după 5 luni, ce acoperă garanția?", "plata": "Pot plăti în rate sau la livrare?",
    "tehnic": "Telefonul nu mai pornește după actualizare.",
}


@dataclass
class Ticket:
    id: int
    topics: list[str]

    @property
    def text(self) -> str:
        return " ".join(TEXTS[t] for t in self.topics) + f" [[meta:{','.join(self.topics)}]]"


def meta(prompt: str) -> list[str]:
    return re.search(r"\[\[meta:([^\]]+)\]\]", prompt).group(1).split(",")


def reply_for(topics: list[str], rng: random.Random, wrong: bool) -> str:
    facts = [f for t in topics for f in FACTS[t]]
    if wrong and facts:
        facts.pop(rng.randrange(len(facts)))
    return "Bună ziua! " + "; ".join(facts) + ". O zi bună!"


def _other(cat: str, rng: random.Random) -> str:
    return rng.choice([c for c in CATS if c != cat])


SKILLS = {
    # clasificare: prima temă a tichetului
    "classify": lambda p, r, w: _other(meta(p)[0], r) if w else meta(p)[0],
    # răspuns pentru categoria primită în prompt (dacă e greșită, faptele sunt greșite)
    "reply": lambda p, r, w: reply_for([re.search(r"categoria: (\w+)", p).group(1)], r, w),
    # un singur apel care face tot: pe tichetele cu două teme, uneori o uită pe a doua
    "single": lambda p, r, w: reply_for(meta(p)[:1] if (w or (len(meta(p)) > 1 and r.random() < 0.5)) else meta(p), r, w),
    # feedback-ul evaluatorului înjumătățește șansa de greșeală, nu o elimină
    "generate": lambda p, r, w: reply_for(meta(p), r, w and ("LIPSEȘTE" not in p or r.random() < 0.5)),
    "evaluate": lambda p, r, w: _evaluate(p, w),
    "plan": lambda p, r, w: "\n".join(meta(p)[:-1] if (w and len(meta(p)) > 1) else meta(p)),
    "worker": lambda p, r, w: reply_for([re.search(r"subproblema: (\w+)", p).group(1)], r, w),
    "synth": lambda p, r, w: "Bună ziua! " + " ".join(re.findall(r"Bună ziua! (.*?)\. O zi bună!", p)) + ". O zi bună!",
}
for c in CATS:
    SKILLS[f"route:{c}"] = lambda p, r, w, c=c: reply_for([c], r, w)


def _evaluate(prompt: str, wrong: bool) -> str:
    needed = [f for t in meta(prompt) for f in FACTS[t]]
    draft = prompt.split("CIORNĂ:", 1)[1]
    missing = [f for f in needed if f not in draft]
    ok = not missing
    if wrong:  # evaluatorul greșește: dă OK unei ciorne incomplete (sau invers)
        ok = not ok
    return "OK" if ok else "LIPSEȘTE: " + ", ".join(missing or ["detalii"])


def complete(ticket: Ticket, answer: str) -> bool:
    return all(f in answer for t in ticket.topics for f in FACTS[t])


def approaches():
    def single_model(model):
        return lambda llm, t, tr: (None, single(llm, t.text, model=model, prompt=lambda s: s, trace=tr))

    def chain_(llm, t, tr):
        steps = [Step("classify", H, lambda s: s, gate=lambda o: o in CATS),
                 Step("reply", S, lambda label: f"categoria: {label} {t.text}")]
        try:
            label = chain(llm, t.text, steps[:1], tr)
            return label, chain(llm, label, steps[1:], tr)
        except GateFailed:
            return None, ""

    def route_(llm, t, tr):
        routes = {c: (O if c in COMPLEX else H, lambda s: s) for c in CATS}
        return route(llm, t.text, classifier_model=H, classify=lambda s: s, parse=str.strip, routes=routes,
                     default="livrare", trace=tr)

    def vote_(llm, t, tr):
        label, _ = vote(llm, t.text, n=5, model=H, prompt=lambda s: s, parse=str.strip, trace=tr)
        return label, llm.complete(f"categoria: {label} {t.text}", model=S, purpose="reply", trace=tr)

    def evalopt(llm, t, tr):
        draft, _, _ = evaluator_optimizer(
            llm, t.text, generator_model=H, generate=lambda task, fb: f"{task} {fb}", evaluator_model=S,
            evaluate=lambda task, d: f"{task} CIORNĂ: {d}", parse_eval=lambda o: (o == "OK", o), trace=tr)
        return None, draft

    def orch(llm, t, tr):
        _, final = orchestrator_workers(
            llm, t.text, orchestrator_model=S, plan=lambda s: s, parse_plan=lambda o: o.split(),
            worker_model=H, worker=lambda task, sub: f"subproblema: {sub} {task}",
            synth=lambda task, rs: "\n".join(rs), trace=tr)
        return None, final

    return {
        "un apel Haiku": single_model(H), "un apel Opus": single_model(O), "chain (Haiku → Sonnet)": chain_,
        "routing (Haiku → Haiku/Opus)": route_, "vot 5×Haiku → Sonnet": vote_,
        "evaluator-optimizer (Haiku ↔ Sonnet)": evalopt, "orchestrator-workers": orch,
    }


def tickets(n: int = 300, seed: int = 1) -> list[Ticket]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        first = rng.choice(CATS)
        topics = [first, _other(first, rng)] if rng.random() < 0.3 else [first]
        out.append(Ticket(i, topics))
    return out


def run(n: int = 300) -> dict[str, dict[str, float]]:
    ts = tickets(n)
    results = {}
    for name, fn in approaches().items():
        llm = SimLLM(SKILLS, seed=42, context_tokens=1200)  # ~1200 tokeni de politici în fiecare apel
        rows = []
        for t in ts:
            tr = Trace()
            label, answer = fn(llm, t, tr)
            rows.append((t, label, answer, tr))
        single_topic = [r for r in rows if len(r[0].topics) == 1]
        multi = [r for r in rows if len(r[0].topics) > 1]
        results[name] = {
            "complete": sum(complete(t, a) for t, _, a, _ in rows) / len(rows),
            "complete_single": sum(complete(t, a) for t, _, a, _ in single_topic) / len(single_topic),
            "complete_multi": sum(complete(t, a) for t, _, a, _ in multi) / len(multi),
            "label_acc": (sum(lbl == t.topics[0] for t, lbl, _, _ in single_topic) / len(single_topic)
                          if rows[0][1] is not None else None),
            "calls": statistics.mean(len(tr.calls) for *_, tr in rows),
            "cost_per_1k": 1000 * statistics.mean(tr.cost for *_, tr in rows),
            "latency_s": statistics.mean(tr.latency_s for *_, tr in rows),
        }
    return results


def main() -> None:
    res = run()
    print("| abordare | răspuns complet | 1 temă | 2 teme | clasificare corectă | apeluri | $ / 1000 tichete | latență medie |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, r in res.items():
        acc = "—" if r["label_acc"] is None else f"{r['label_acc']:.0%}"
        print(f"| {name} | {r['complete']:.0%} | {r['complete_single']:.0%} | {r['complete_multi']:.0%} | {acc} | {r['calls']:.1f} | "
              f"{r['cost_per_1k']:.3f} | {r['latency_s']:.2f} s |")


if __name__ == "__main__":
    main()
