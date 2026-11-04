"""Question and answer generation from a state-store snapshot.

``snapshot_facts(store)`` copies what the summary says into plain data.
``make_examples(facts, summary, rng)`` turns one snapshot into several
(question, answer, checks) examples. Every answer uses only facts that are in
the summary text. ``checks`` lists strings an answer must contain and must not
contain; ``evaluate.py`` uses them to score model answers.

``answer(facts, question)`` is a deterministic fallback answerer (keyword
intent matching plus the same templates). The bot uses it when no model is
loaded, and evaluation uses it as a baseline.
"""

from __future__ import annotations

import random
import re
from typing import Any

from monitor.state import StateStore
from monitor.summary import build_summary, describe, fmt_num, fmt_time, spike_groups

SYSTEM_PROMPT = (
    "You answer questions about a monitored sensor system. You are given a status summary written by "
    "automatic detectors. Answer only from the summary: say what is abnormal, since when, and how large, "
    "using the channel names and times exactly as written. If the summary does not contain the answer "
    "(causes, predictions, advice, or channels not listed), say that the summary does not say. Be brief."
)

UNKNOWN_CHANNELS = ["helium_level", "beam_position_x", "dewar_pressure", "Labjack.coil_current",
                    "magnet_quench_signal", "cryostat_vacuum", "QT.fc509_ai", "laser_power"]

KIND_WORDS = {
    "drift_up": "running high", "drift_down": "running low", "stuck": "stuck", "noisy": "noisy",
    "missing": "missing readings", "decoupled": "not tracking its source",
}


def snapshot_facts(store: StateStore) -> dict[str, Any]:
    """The facts the summary states, as plain data (times as HH:MM strings, as in the summary)."""
    now = store.now
    abnormal = []
    for ch in sorted(store.abnormal_channels(), key=lambda c: min(x.since for x in c.active.values())):
        conds = []
        for c in ch.active.values():
            conds.append({
                "kind": c.kind, "since": fmt_time(c.since, now), "text": describe(ch, c, now),
                "deviation": None if c.magnitude is None else fmt_num(abs(c.magnitude)),
                "direction": None if c.magnitude is None else ("above" if c.magnitude >= 0 else "below"),
                "sigma": None if c.magnitude_sigma is None else f"{abs(c.magnitude_sigma):.1f}",
                "source": c.detail.get("source"), "value": c.detail.get("value"),
            })
        abnormal.append({"name": ch.display_name, "normal": fmt_num(ch.cal.target), "conditions": conds})
    spikes = spike_groups(store)[:6]
    feeds_down = [{"name": f.name, "since": fmt_time(f.outage.since, now)}
                  for f in store.feeds.values() if f.outage is not None]
    return {
        "now": now.strftime("%Y-%m-%d %H:%M") if now else None,
        "n_channels": len(store.channels),
        "abnormal": abnormal,
        "normal": [c.display_name for c in store.channels.values() if not c.active],
        "spikes": spikes,
        "feeds": sorted(store.feeds),
        "feeds_down": feeds_down,
    }


def summary_of(store: StateStore) -> tuple[str, dict[str, Any]]:
    return build_summary(store), snapshot_facts(store)


# ---------------------------------------------------------------------------
# Answer templates
# ---------------------------------------------------------------------------
def _join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _channel_sentence(a: dict[str, Any]) -> str:
    return f"{a['name']} is " + "; it is also ".join(c["text"] for c in a["conditions"]) + "."


YES_NO = re.compile(r"^(is|are|any|anything|should|do|does|has|have|can)\b", re.I)
ALL_OK = re.compile(r"everything (ok|okay|fine|good|normal)", re.I)


def _overall(f: dict[str, Any], rng: random.Random, question: str = "") -> str:
    parts = []
    yes_no = bool(YES_NO.match(question.strip()))
    all_ok = bool(ALL_OK.search(question))
    if not f["abnormal"]:
        if all_ok:
            opener = "Yes, everything looks normal."
        elif yes_no:
            opener = "No, everything looks normal."
        else:
            opener = rng.choice(["Nothing is abnormal right now.", "All clear: no channel is abnormal."])
        parts.append(f"{opener} All {f['n_channels']} monitored channels are normal as of {f['now']} UTC.")
    else:
        n = len(f["abnormal"])
        names = _join([a["name"] for a in f["abnormal"]])
        lead = "No. " if all_ok else ("Yes. " if yes_no else "")
        parts.append(f"{lead}{n} of {f['n_channels']} channels {'is' if n == 1 else 'are'} abnormal: {names}.")
        parts += [_channel_sentence(a) for a in f["abnormal"]]
    if f["feeds_down"]:
        parts.append(" ".join(f"{d['name']} has sent no rows since {d['since']}." for d in f["feeds_down"]))
    if f["spikes"]:
        parts.append("Recent spikes: " + _join([s["text"] for s in f["spikes"]]) + ".")
    return " ".join(parts)


def _about(f: dict[str, Any], name: str) -> str:
    for a in f["abnormal"]:
        if a["name"] == name:
            return f"{_channel_sentence(a)} Its normal level is {a['normal']}."
    if name in f["normal"]:
        sp = [s for s in f["spikes"] if s["name"] == name]
        extra = ""
        if sp:
            g = sp[-1]
            extra = (f" It had a spike at {g['time']} ({g['sigma']})." if g["count"] == 1
                     else f" It had {g['count']} spikes recently, the last at {g['time']} (up to {g['sigma']}).")
        return f"{name} is normal right now.{extra}"
    return f"The summary does not mention {name}, so I can't say anything about it."


def _since(f: dict[str, Any], a: dict[str, Any]) -> str:
    c = a["conditions"][0]
    return f"{a['name']} has been {KIND_WORDS.get(c['kind'], c['kind'])} since {c['since']}."


def _how_big(a: dict[str, Any]) -> str | None:
    for c in a["conditions"]:
        if c["deviation"] is not None and c["kind"] in ("drift_up", "drift_down"):
            return (f"{a['name']} is about {c['deviation']} {c['direction']} its normal level of {a['normal']}, "
                    f"which is {c['sigma']} sigma.")
    return None


def _normal_list(f: dict[str, Any]) -> str:
    if not f["normal"]:
        return "None of the channels are normal right now."
    if len(f["normal"]) > 12:
        return f"{len(f['normal'])} channels are normal: {_join(f['normal'])}."
    return f"These channels are normal: {_join(f['normal'])}."


def _feeds(f: dict[str, Any]) -> str:
    if not f["feeds"]:
        return "The summary has no feed information."
    if not f["feeds_down"]:
        return "Yes, all feeds are reporting."
    return "No. " + " ".join(f"{d['name']} has sent no rows since {d['since']}." for d in f["feeds_down"])


def _spikes(f: dict[str, Any]) -> str:
    if not f["spikes"]:
        return "No spikes in the last 60 minutes."
    return "Yes: " + _join([s["text"] for s in f["spikes"]]) + "."


def _why(f: dict[str, Any], a: dict[str, Any] | None, question: str = "") -> str:
    if re.search(r"should i do|fix|what do i do|how do i", question.lower()):
        base = ("The summary does not say what to do. It only reports what the detectors found; "
                "decisions are up to the people running the experiment.")
    else:
        base = "The summary does not say why. The detectors only report what changed, not the cause."
    if a is not None:
        base += f" What it does say: {_channel_sentence(a)}"
    return base


def _predict(f: dict[str, Any], name: str) -> str:
    return (f"I can't predict that. The summary only describes the current state"
            + (f": {_about(f, name)}" if name else "."))


# ---------------------------------------------------------------------------
# Example generation
# ---------------------------------------------------------------------------
Q_OVERALL = ["Is anything abnormal right now?", "Any problems?", "What's wrong?", "Status?",
             "Give me a status report.", "Is everything ok?", "anything weird going on?",
             "What is abnormal at the moment?", "Should I be worried about anything?", "how are things looking"]
Q_ABOUT = ["What's going on with {n}?", "Is {n} ok?", "How is {n} doing?", "Tell me about {n}.",
           "any issues with {n}?", "status of {n}?"]
Q_SINCE = ["Since when has {n} been {w}?", "When did the problem with {n} start?", "How long has {n} been {w}?"]
Q_BIG = ["How far off is {n}?", "How big is the deviation on {n}?", "How much has {n} moved?"]
Q_NORMAL = ["Which channels are normal?", "What's working fine?", "Which channels are ok?"]
Q_FEEDS = ["Are all feeds reporting?", "Is any feed down?", "Is data still coming in?"]
Q_SPIKES = ["Any spikes recently?", "Were there any spikes?", "Any glitches in the last hour?"]
Q_WHY = ["Why is {n} {w}?", "What caused the problem with {n}?", "What should I do about {n}?"]
Q_PREDICT = ["What will {n} be in an hour?", "Will {n} recover soon?", "Is {n} going to get worse?"]
Q_UNKNOWN = ["What about {n}?", "Is {n} ok?", "How is {n} doing?"]


# Phrasings used only for the held-out test file (test_holdout.jsonl), never in training.
# They check that the model handles wording it has not seen.
HOLDOUT = {
    "OVERALL": ["yo, anything off?", "Give me the rundown.", "Are there any alarms at the moment?",
                "is something broken", "Quick health check please."],
    "ABOUT": ["anything off with {n}?", "{n} looking alright?", "what's the deal with {n}",
              "can you check {n} for me?"],
    "SINCE": ["what time did {n} start acting up?", "When did {n} go {w}?"],
    "BIG": ["by how much is {n} off?", "what's the size of the change on {n}?"],
    "NORMAL": ["what's healthy right now?", "List the channels with no problems."],
    "FEEDS": ["did any of the feeds stop?", "are we still getting rows from every table?"],
    "SPIKES": ["seen any blips lately?", "Were there spikes in the last hour?"],
    "WHY": ["what's causing {n} to be {w}?", "how do I fix {n}?"],
    "PREDICT": ["is {n} going to come back to normal?", "where will {n} be tomorrow?"],
    "UNKNOWN": ["how's {n} looking?", "anything off with {n}?"],
}
TRAIN = {"OVERALL": Q_OVERALL, "ABOUT": Q_ABOUT, "SINCE": Q_SINCE, "BIG": Q_BIG, "NORMAL": Q_NORMAL,
         "FEEDS": Q_FEEDS, "SPIKES": Q_SPIKES, "WHY": Q_WHY, "PREDICT": Q_PREDICT, "UNKNOWN": Q_UNKNOWN}

def _checks(include: list[str], exclude: list[str] | None = None, refusal: bool = False) -> dict[str, Any]:
    return {"must_include": [s for s in include if s], "must_not_include": exclude or [], "refusal": refusal}


def make_examples(facts: dict[str, Any], summary: str, rng: random.Random, max_examples: int = 6,
                  holdout: bool = False) -> list[dict]:
    """A handful of varied examples for one snapshot. ``holdout`` uses the held-out phrasings."""
    Q = HOLDOUT if holdout else TRAIN
    ab = facts["abnormal"]
    abnormal_names = [a["name"] for a in ab]
    out: list[dict] = []

    def add(intent, q, a, checks):
        out.append({"intent": intent, "question": q, "answer": a, "checks": checks})

    q_overall = rng.choice(Q["OVERALL"])
    add("overall", q_overall, _overall(facts, rng, q_overall),
        _checks(abnormal_names + [d["name"] for d in facts["feeds_down"]],
                [n for n in facts["normal"] if n not in abnormal_names][:0]))
    if ab:
        a = rng.choice(ab)
        add("about_abnormal", rng.choice(Q["ABOUT"]).format(n=a["name"]), _about(facts, a["name"]),
            _checks([a["name"], a["conditions"][0]["since"]]))
        c = a["conditions"][0]
        add("since", rng.choice(Q["SINCE"]).format(n=a["name"], w=KIND_WORDS.get(c["kind"], "abnormal")),
            _since(facts, a), _checks([c["since"]]))
        big = _how_big(a)
        if big:
            add("magnitude", rng.choice(Q["BIG"]).format(n=a["name"]), big,
                _checks([a["conditions"][0]["deviation"] if a["conditions"][0]["kind"].startswith("drift") else "",
                         a["normal"]]))
        q_why = rng.choice(Q["WHY"]).format(n=a["name"], w=KIND_WORDS.get(c["kind"], "abnormal"))
        add("why", q_why, _why(facts, a, q_why), _checks(["does not say"], refusal=True))
    if facts["normal"]:
        n = rng.choice(facts["normal"])
        add("about_normal", rng.choice(Q["ABOUT"]).format(n=n), _about(facts, n), _checks([n, "normal"]))
    if rng.random() < 0.5:
        add("normal_list", rng.choice(Q["NORMAL"]), _normal_list(facts),
            _checks(facts["normal"]))
    if facts["feeds"] and (facts["feeds_down"] or rng.random() < 0.3):
        add("feeds", rng.choice(Q["FEEDS"]), _feeds(facts), _checks([d["name"] for d in facts["feeds_down"]]))
    if facts["spikes"] or rng.random() < 0.2:
        add("spikes", rng.choice(Q["SPIKES"]), _spikes(facts), _checks([s["name"] for s in facts["spikes"]]))
    if rng.random() < 0.35:
        name = rng.choice(abnormal_names + facts["normal"])
        add("predict", rng.choice(Q["PREDICT"]).format(n=name), _predict(facts, name),
            _checks(["can't predict"], refusal=True))
    if rng.random() < 0.3:
        u = rng.choice([x for x in UNKNOWN_CHANNELS if x not in abnormal_names + facts["normal"]])
        add("unknown_channel", rng.choice(Q["UNKNOWN"]).format(n=u), _about(facts, u),
            _checks(["does not mention"], refusal=True))
    rng.shuffle(out)
    first = [e for e in out if e["intent"] == "overall"]
    rest = [e for e in out if e["intent"] != "overall"]
    return (first + rest)[:max_examples]


def to_chat(summary: str, question: str, answer: str | None = None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Summary:\n{summary}\n\nQuestion: {question}"}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


# ---------------------------------------------------------------------------
# Deterministic fallback answerer
# ---------------------------------------------------------------------------
def _find_channel(facts: dict[str, Any], q: str) -> str | None:
    ql = q.lower()
    names = [a["name"] for a in facts["abnormal"]] + facts["normal"]
    hits = [n for n in names if n.lower() in ql]
    return max(hits, key=len) if hits else None


def answer(facts: dict[str, Any], question: str) -> str:
    """Keyword-based answer from the same templates. Used as a fallback and a baseline."""
    q = question.lower()
    name = _find_channel(facts, question)
    a = next((x for x in facts["abnormal"] if x["name"] == name), None)
    if re.search(r"\bwhy\b|cause|should i do|fix", q):
        return _why(facts, a, question)
    if re.search(r"will |going to|predict|in an hour|soon", q):
        return _predict(facts, name or "")
    if "spike" in q or "glitch" in q:
        return _spikes(facts)
    if "feed" in q or "coming in" in q or "reporting" in q:
        return _feeds(facts)
    if name is None:
        # A name-like token (with _ or .) that is not in the summary: an unknown channel.
        unknown = re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[_.][A-Za-z0-9]+)+", question)
        if unknown:
            return _about(facts, unknown[0])
        if "abnormal" not in q and re.search(r"which channels|working fine|\bnormal\b", q):
            return _normal_list(facts)
        return _overall(facts, random.Random(0), question)
    if a is not None and re.search(r"since|when|how long", q):
        words = (("high", "drift_up"), ("low", "drift_down"), ("stuck", "stuck"), ("nois", "noisy"),
                 ("missing", "missing"), ("track", "decoupled"))
        asked = next((k for word, k in words if word in q), None)
        conds = [c for c in a["conditions"] if c["kind"] == asked] or a["conditions"]
        return _since(facts, {**a, "conditions": conds})
    if a is not None and re.search(r"how far|how big|how much|deviation", q):
        return _how_big(a) or _about(facts, name)
    return _about(facts, name)
