# 🧠 agent-patterns

Cele cinci pattern-uri din [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents),
scrise de mână, cu **costul, latența și numărul de apeluri contabilizate identic**, ca să poată fi comparate pe aceeași problemă.

Întrebarea pe care o lămurește: *„Workflow sau agent? Când merită 6 apeluri în loc de 1?”*

> Pașii 2 și 4 din planul meu de studiu: „pot explica diferența dintre un workflow și un agent” și „când NU e nevoie de multi-agent”.

![CI](https://github.com/cs1lyuha/agent-patterns/actions/workflows/ci.yml/badge.svg)

## Pattern-urile

| pattern | ce face | când merită | cost |
|---|---|---|---|
| **prompt chaining** | pași secvențiali, cu **porți** programatice între ei | pași clari, fiecare verificabil | n apeluri, latențe adunate |
| **routing** | un apel ieftin clasifică, apoi cererea merge la modelul / promptul potrivit | intrări de tipuri diferite, cu costuri diferite | +1 apel ieftin |
| **sectioning** | subprobleme independente, în paralel | părți care nu depind una de alta | n apeluri, **latența celui mai lent** |
| **voting** | același apel de n ori, majoritate + acord | decizii la care o greșeală costă mult | n × apelul |
| **orchestrator-workers** | un model planifică subproblemele (dinamic), workerii le rezolvă, apoi o sinteză | nu știi dinainte câte subprobleme sunt | plan + n + sinteză |
| **evaluator-optimizer** | generator ↔ evaluator, până trece | criterii clare de calitate, iar feedback-ul ajută | 2 apeluri per rundă |

Latența se calculează pe un **ceas virtual**: apelurile secvențiale se adună, iar cele paralele contează cât maximul lor (`patterns/llm.py:parallel`).

## Benchmark: 300 de tichete de suport

`python -m bench.support`. 30% dintre tichete au **două teme** („Vreau să returnez produsul. Pot plăti în rate?”).
Un răspuns e *complet* dacă are toate faptele necesare (sume, termene) pentru toate temele.

| abordare | răspuns complet | 1 temă | 2 teme | clasificare corectă | apeluri | $ / 1000 tichete | latență |
|---|---:|---:|---:|---:|---:|---:|---:|
| un apel Haiku | 66% | 77% | 39% | — | 1.0 | 0.13 | 0.44 s |
| un apel Opus | 83% | 96% | 50% | — | 1.0 | 5.09 | 1.82 s |
| chain (Haiku → Sonnet) | 51% | 70% | 0% | 76% | 2.0 | 2.67 | 1.42 s |
| routing (Haiku → Haiku/Opus) | 48% | 66% | 0% | 76% | 2.0 | 2.33 | 1.41 s |
| vot 5×Haiku → Sonnet | 63% | 88% | 0% | **98%** | 6.0 | 3.16 | 1.42 s |
| **evaluator-optimizer** (Haiku ↔ Sonnet) | **98%** | 97% | **99%** | — | 2.7 | 3.47 | 1.82 s |
| orchestrator-workers | 75% | 82% | 57% | — | 3.3 | 5.15 | 2.39 s |

> ⚠️ **Simulare, nu un model real.** `SimLLM` greșește independent, cu rata fixată per model: Haiku 20%, Sonnet 10%, Opus 5%.
> Fiecare apel include ~1200 de tokeni de context; un prompt „care face tot” uită a doua temă în 50% din cazuri; feedback-ul evaluatorului înjumătățește greșelile.
> Tabelul arată deci **consecințele structurale** ale fiecărui pattern: cum se compun erorile, câte apeluri, ce latență. Nu e un clasament al modelelor.
> Același cod rulează pe Claude cu `ClaudeLLM`.

### Ce se vede

1. **Routing-ul și chaining-ul presupun o singură categorie.** Pe tichetele cu două teme ajung la **0%**: clasificatorul alege o temă, iar a doua se pierde. Un pattern e corect doar dacă forma problemei se potrivește cu el.
2. **Votul repară clasificarea, nu și răspunsul:** 76% → 98% clasificare corectă, cu latență egală (apelurile sunt paralele) și de 3 ori mai multe apeluri. Completitudinea rămâne însă limitată de pasul de după.
3. **Erorile se înmulțesc în lanț:** clasificare 80% × răspuns 90% ≈ 72%. Un pipeline de modele „bune” iese mai slab decât fiecare dintre ele.
4. **Evaluator-optimizer** ajunge la 98% cu un generator ieftin, la **68% din costul unui apel Opus**. Merge pentru că „are toate faptele?” e un criteriu verificabil. Fără un criteriu clar, pattern-ul doar adaugă cost.
5. **Orchestrator-workers** are cea mai mare latență (plan → workeri → sinteză) și, pe tichete simple, nu aduce nimic față de un singur apel. Merită când numărul de subprobleme chiar variază.

**Regula din articol se confirmă:** începi cu un singur apel bun și adaugi structură doar când o măsurătoare arată de ce.

## Cod

```python
from patterns.llm import ClaudeLLM, Trace
from patterns.workflows import evaluator_optimizer

trace = Trace()
draft, rounds, ok = evaluator_optimizer(
    ClaudeLLM(), ticket_text,
    generator_model="claude-haiku-5-5", generate=lambda task, feedback: f"Răspunde la tichet: {task}\n{feedback}",
    evaluator_model="claude-sonnet-5-5", evaluate=lambda task, d: f"Verifică dacă răspunsul ... Tichet: {task}\nCiornă: {d}",
    parse_eval=lambda out: (out.strip() == "OK", out), trace=trace)
print(rounds, ok, f"${trace.cost:.4f}", f"{trace.latency_s:.1f}s")
```

## Pornire

```bash
git clone https://github.com/cs1lyuha/agent-patterns.git && cd agent-patterns
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
pytest -q                   # 16 teste, inclusiv relațiile din tabel
python -m bench.support     # tabelul de mai sus
```

## Structura

| Fișier | Rol |
|---|---|
| `patterns/llm.py` | `Trace` (apeluri, tokeni, cost, latență virtuală), `parallel`, `SimLLM` determinist pe thread-uri, `ClaudeLLM` |
| `patterns/workflows.py` | `single`, `chain` (cu porți), `route`, `sectioning`, `vote`, `orchestrator_workers`, `evaluator_optimizer` |
| `bench/support.py` | scenariul cu tichete, ipotezele simulării, tabelul |
