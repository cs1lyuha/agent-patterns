import pytest

from patterns.llm import SimLLM, Trace
from patterns.workflows import (GateFailed, Step, chain, evaluator_optimizer, orchestrator_workers, route,
                                sectioning, vote)

H, S, O = "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"


def echo_llm(**skills):
    base = {"echo": lambda p, r, w: p.upper()}
    base.update(skills)
    return SimLLM(base, error_rate={H: 0.0, S: 0.0, O: 0.0})


def test_chain_runs_steps_in_order_and_sums_latency():
    llm = echo_llm(a=lambda p, r, w: p + "-a", b=lambda p, r, w: p + "-b")
    t = Trace()
    out = chain(llm, "x", [Step("a", H, lambda s: s), Step("b", S, lambda s: s)], t)
    assert out == "x-a-b" and [c.purpose for c in t.calls] == ["a", "b"]
    assert t.latency_s == pytest.approx(sum(c.latency_s for c in t.calls))


def test_chain_gate_stops_early():
    llm = echo_llm(a=lambda p, r, w: "invalid", b=lambda p, r, w: "never")
    t = Trace()
    with pytest.raises(GateFailed):
        chain(llm, "x", [Step("a", H, lambda s: s, gate=lambda o: o != "invalid"), Step("b", O, lambda s: s)], t)
    assert len(t.calls) == 1  # pasul scump nu a mai rulat


def test_route_picks_model_per_label():
    llm = echo_llm(classify=lambda p, r, w: "tehnic" if "eroare" in p else "simplu",
                   **{"route:tehnic": lambda p, r, w: "detaliat", "route:simplu": lambda p, r, w: "scurt"})
    routes = {"tehnic": (O, lambda s: s), "simplu": (H, lambda s: s)}
    t = Trace()
    label, out = route(llm, "am o eroare", classifier_model=H, classify=lambda s: s, parse=str.strip,
                       routes=routes, default="simplu", trace=t)
    assert (label, out, t.calls[1].model) == ("tehnic", "detaliat", O)


def test_route_unknown_label_falls_back_to_default():
    llm = echo_llm(classify=lambda p, r, w: "???", **{"route:simplu": lambda p, r, w: "ok"})
    label, _ = route(llm, "x", classifier_model=H, classify=lambda s: s, parse=str.strip,
                     routes={"simplu": (H, lambda s: s)}, default="simplu", trace=Trace())
    assert label == "simplu"


def test_parallel_latency_is_max_not_sum():
    llm = echo_llm(scurt=lambda p, r, w: "a", lung=lambda p, r, w: "b" * 400)
    t = Trace()
    out = sectioning(llm, "x", {"scurt": (H, lambda s: s), "lung": (O, lambda s: s)}, t)
    assert out == {"scurt": "a", "lung": "b" * 400}
    assert t.latency_s == pytest.approx(max(c.latency_s for c in t.calls))


def test_vote_majority_and_agreement():
    llm = SimLLM({"classify": lambda p, r, w: "gresit" if w else "corect"}, error_rate={H: 0.3, S: 0, O: 0}, seed=3)
    label, agreement = vote(llm, "tichet", n=7, model=H, prompt=lambda s: s, parse=str.strip, trace=Trace())
    assert label == "corect" and 0.5 < agreement <= 1.0


def test_vote_is_deterministic_despite_threads():
    def once():
        llm = SimLLM({"classify": lambda p, r, w: "B" if w else "A"}, error_rate={H: 0.45, S: 0, O: 0}, seed=9)
        return [vote(llm, f"t{i}", n=5, model=H, prompt=lambda s: s, parse=str.strip, trace=Trace()) for i in range(20)]
    assert once() == once()


def test_orchestrator_workers():
    llm = echo_llm(plan=lambda p, r, w: "livrare\nretur\n", worker=lambda p, r, w: "răspuns " + p.split("|")[1],
                   synth=lambda p, r, w: p)
    t = Trace()
    subs, final = orchestrator_workers(
        llm, "două întrebări", orchestrator_model=S, plan=lambda s: s, parse_plan=lambda o: o.split(),
        worker_model=H, worker=lambda task, sub: f"{task}|{sub}", synth=lambda task, rs: " / ".join(rs), trace=t)
    assert subs == ["livrare", "retur"] and final == "răspuns livrare / răspuns retur"
    assert [c.purpose for c in t.calls].count("worker") == 2


def test_evaluator_optimizer_iterates_until_ok():
    llm = echo_llm(generate=lambda p, r, w: "draft+fix" if "lipsește" in p else "draft",
                   evaluate=lambda p, r, w: "OK" if "fix" in p else "lipsește termenul")
    draft, rounds, ok = evaluator_optimizer(
        llm, "răspunde", generator_model=H, generate=lambda task, fb: f"{task} {fb}", evaluator_model=S,
        evaluate=lambda task, d: d, parse_eval=lambda o: (o == "OK", o), trace=Trace())
    assert (draft, rounds, ok) == ("draft+fix", 2, True)


def test_evaluator_optimizer_gives_up():
    llm = echo_llm(generate=lambda p, r, w: "rău", evaluate=lambda p, r, w: "nu")
    _, rounds, ok = evaluator_optimizer(llm, "t", generator_model=H, generate=lambda t, f: t, evaluator_model=S,
                                        evaluate=lambda t, d: d, parse_eval=lambda o: (False, o), trace=Trace(),
                                        max_rounds=2)
    assert (rounds, ok) == (2, False)
