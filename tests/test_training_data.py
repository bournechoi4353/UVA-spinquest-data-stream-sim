"""Question/answer generation, the summary parser, and the evaluation checks."""

import random

import pytest

from monitor.engine import Coupling, Engine
from sim import replay
from training.evaluate import check
from training.qa import SYSTEM_PROMPT, answer, make_examples, snapshot_facts, to_chat
from training.summary_parse import facts_from_summary
from monitor.summary import build_summary


@pytest.fixture(scope="module")
def store():
    eng = Engine.from_healthy(replay("configs/healthy.yaml"),
                              couplings=[Coupling("nmr_signal", "polarization", 0.15)])
    for t, ev in replay("configs/mixed_demo.yaml"):
        eng.step(t, ev)
        if t.strftime("%H:%M:%S") == "12:20:00":
            break
    return eng.store


def test_parser_recovers_the_facts_from_the_summary_text(store):
    summary = build_summary(store)
    direct = snapshot_facts(store)
    parsed = facts_from_summary(summary)
    assert parsed["now"] == direct["now"] and parsed["n_channels"] == direct["n_channels"]
    assert [a["name"] for a in parsed["abnormal"]] == [a["name"] for a in direct["abnormal"]]
    assert parsed["normal"] == direct["normal"]
    assert [c["since"] for a in parsed["abnormal"] for c in a["conditions"]] == \
           [c["since"] for a in direct["abnormal"] for c in a["conditions"]]
    assert [s["name"] for s in parsed["spikes"]] == [s["name"] for s in direct["spikes"]]


def test_examples_are_grounded_in_the_summary(store):
    summary = build_summary(store)
    facts = snapshot_facts(store)
    rng = random.Random(0)
    for _ in range(30):
        for ex in make_examples(facts, summary, rng):
            res = check(ex["answer"], summary, ex["question"], ex["checks"])
            assert res["pass"], (ex["intent"], ex["question"], ex["answer"])
            assert not res["hallucinated"], (ex["question"], ex["answer"], res["bad"])


def test_refusals_for_causes_predictions_and_unknown_channels(store):
    facts = snapshot_facts(store)
    assert "does not say" in answer(facts, "Why is polarization running low?")
    assert "can't predict" in answer(facts, "Will polarization recover soon?")
    assert "does not mention" in answer(facts, "Is helium_level ok?")
    assert "what to do" in answer(facts, "What should I do about rf_power?")


def test_overall_answer_lists_every_abnormal_channel(store):
    facts = snapshot_facts(store)
    text = answer(facts, "Is anything abnormal right now?")
    for a in facts["abnormal"]:
        assert a["name"] in text
    assert text.startswith("Yes.")
    assert not answer(facts, "What's wrong?").startswith("Yes.")


def test_chat_format_has_system_user_and_assistant():
    msgs = to_chat("Status at ...", "Any problems?", "No.")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"].startswith("Summary:\n") and "\n\nQuestion: Any problems?" in msgs[1]["content"]


def test_check_flags_hallucinated_names_and_times():
    summary = "Status at 2026-01-01 10:00 UTC. 1 channels monitored, 0 abnormal.\nNormal: rf_power."
    ok = check("rf_power is normal.", summary, "Is rf_power ok?", {"must_include": ["rf_power"],
                                                                    "must_not_include": [], "refusal": False})
    bad = check("beam_current has been low since 09:12.", summary, "Is rf_power ok?",
                {"must_include": [], "must_not_include": [], "refusal": False})
    assert ok["pass"] and not ok["hallucinated"]
    assert bad["hallucinated"] and set(bad["bad"]) == {"09:12", "beam_current"}
