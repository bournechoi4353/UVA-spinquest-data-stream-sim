"""The hard question sets: every ideal answer passes its own checks, and the stricter checks work."""

import json
import random

from training.evaluate import check
from training.hard_set import build, build_noisy, items, scenarios, scramble


def test_every_ideal_answer_passes_its_own_checks(tmp_path):
    n = build(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "test_hard.jsonl").read_text(encoding="utf-8").splitlines()]
    assert n == len(rows) == len(items()) >= 60
    for r in rows:
        user = r["messages"][1]["content"]
        q = user.split("\n\nQuestion: ", 1)[1]
        ideal = r["messages"][2]["content"]
        if ideal.startswith("("):
            continue                                         # placeholder for a long list
        res = check(ideal, user, q, r["checks"])
        assert res["pass"], (r["intent"], q, res)


def test_scenarios_cover_the_edge_cases():
    s = scenarios()
    assert "12 abnormal" in s["storm"] and "and spikes on 2 more channels" in s["storm"]
    assert "Labjack has sent no rows since 13:40" in s["storm"]
    assert "QT has sent no rows since 13:58" in s["outage"] and "0 abnormal" in s["outage"]
    assert s["nodata"].startswith("Status: no data yet.") and "Normal:" not in s["nodata"]
    assert "since 2026-05-01 22:10" in s["old"]
    assert "-41030.7" in s["numbers"] and "5.2e-05" in s["numbers"]


def test_stricter_checks():
    summary = "Status at 2026-05-03 14:20 UTC. 29 channels monitored, 1 abnormal.\n- PT501 pressure: running high"
    c = {"must_include": [], "must_include_any": [["running high", "high"]], "must_not_include": [],
         "refusal": False, "no_refusal": True, "strict_numbers": True}
    assert check("PT501 pressure is running high since 14:20.", summary, "q", c)["pass"]
    bad = check("PT501 pressure is running high, about 7.7 above normal.", summary, "q", c)
    assert not bad["pass"] and "strict_numbers" in bad["failed"]
    ref = check("The summary does not say.", summary, "q", c)
    assert not ref["pass"] and "no_refusal" in ref["failed"]


def test_scrambled_questions_keep_their_checks(tmp_path):
    src = tmp_path / "src.jsonl"
    row = {"messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": "Summary:\nS\n\nQuestion: Is PT501 pressure normal?"},
                        {"role": "assistant", "content": "a"}], "intent": "about_normal",
           "checks": {"must_include": ["PT501 pressure"], "must_not_include": [], "refusal": False}}
    src.write_text(json.dumps(row) + "\n")
    assert build_noisy(src, tmp_path) == 1
    out = json.loads((tmp_path / "test_noisy.jsonl").read_text().splitlines()[0])
    assert out["checks"] == row["checks"] and out["intent"] == "about_normal"
    q = out["messages"][1]["content"].split("Question: ", 1)[1]
    assert q != "Is PT501 pressure normal?"
    rng = random.Random(0)
    assert len({scramble("Is PT501 pressure normal?", rng, ["PT501 pressure"]) for _ in range(20)}) > 10
