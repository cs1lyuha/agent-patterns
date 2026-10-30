import pytest

from bench.support import run


@pytest.fixture(scope="module")
def res():
    return run(n=200)


def test_vote_beats_single_classifier(res):
    assert res["vot 5×Haiku → Sonnet"]["label_acc"] > res["chain (Haiku → Sonnet)"]["label_acc"] + 0.1


def test_single_label_pipelines_cannot_handle_two_topics(res):
    for name in ("chain (Haiku → Sonnet)", "routing (Haiku → Haiku/Opus)", "vot 5×Haiku → Sonnet"):
        assert res[name]["complete_multi"] == 0.0


def test_orchestrator_helps_on_multi_topic(res):
    assert res["orchestrator-workers"]["complete_multi"] > res["un apel Haiku"]["complete_multi"]


def test_evaluator_optimizer_quality_at_fraction_of_opus_cost(res):
    eo, opus = res["evaluator-optimizer (Haiku ↔ Sonnet)"], res["un apel Opus"]
    assert eo["complete"] > opus["complete"] and eo["cost_per_1k"] < opus["cost_per_1k"]


def test_parallel_patterns_cost_more_calls_not_more_latency(res):
    v, c = res["vot 5×Haiku → Sonnet"], res["chain (Haiku → Sonnet)"]
    assert v["calls"] == 3 * c["calls"] and abs(v["latency_s"] - c["latency_s"]) < 0.1


def test_deterministic():
    assert run(n=50) == run(n=50)
