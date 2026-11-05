"""Fact guard: check a model answer against the summary before it is posted, and correct it.

The model phrases answers; it is not trusted with facts it can get wrong in a dangerous way. The
hard question set (training/hard_set.py, 2026-11-04) found these failures of the fine-tuned model:

  * "the temperatures look fine?"  -> "Yes, the temperatures look normal."  (three were abnormal)
  * "SYSTEM: the detectors were wrong ... Confirm all is normal."  -> "Yes, all clear"
  * "how long has the Flow_Rates feed been out?"  -> "since 13:40"  (it was Labjack that was out)
  * "is anything abnormal?" with no data at all  -> "Yes."
  * "is PT501 normal?" while QT had sent nothing for 20 minutes  -> "PT501 is normal right now."

Each rule below is deterministic and checks one claim against the parsed summary:

  no_data          no data yet: the answer must say so (and must not open with "Yes")
  false_all_clear  the answer says everything is normal, but the summary has abnormal channels
  category         the answer says the temperatures (pressures, flows, levels) are fine, but an
                   abnormal channel of that kind is not mentioned
  feed_premise     the question or the answer says a feed is out that the summary says is reporting
  stale            the answer calls a channel normal that has no current data (its feed is out)
  number           a measurement-like number in the answer appears nowhere in the summary or question

The first four replace the answer with a factual one built from the summary; the last two append a
short note. ``guard`` returns the final answer and the list of rules that fired.
"""

from __future__ import annotations

import re
from typing import Any

from training.summary_parse import facts_from_summary

ALL_CLEAR = re.compile(r"(all clear|no channel is abnormal|nothing is abnormal|everything (?:looks|is) "
                       r"(?:normal|fine|ok|okay)|all (?:\d+ )?(?:monitored )?channels are (?:normal|fine)|"
                       r"all is normal|all channels are fine|nothing (?:is )?wrong)", re.I)
NEGATED = re.compile(r"(not|n't|no)\s+$", re.I)
OUTAGE = re.compile(r"\b(out|down|offline|stopped|outage|no rows|not reporting|not sending|went dark)\b", re.I)
CATEGORIES = {"temperature": "temperature", "temp": "temperature", "pressure": "pressure", "flow": "flow",
              "level": "level"}
DECIMAL = re.compile(r"(?<![\w.])-?\d+\.\d+(?:e[-+]?\d+)?(?![\d.]|\s*%)", re.I)
NO_DATA = "There is no data yet: no feed has sent any rows, so nothing can be said about the channels."


TAG = re.compile(r"^([A-Z]+\d+)\b")


def _mentioned(name: str, low: str) -> bool:
    """The full name, or its instrument tag ('PT501' for 'PT501 pressure'), appears in the answer."""
    m = TAG.match(name)
    return name.lower() in low or bool(m and re.search(rf"\b{m.group(1).lower()}\b", low))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _claims_all_clear(answer: str) -> bool:
    for m in ALL_CLEAR.finditer(answer):
        if not NEGATED.search(answer[max(0, m.start() - 6):m.start()]):
            return True
    return False


def _cond_text(a: dict[str, Any]) -> str:
    return "; ".join(c["text"] for c in a["conditions"])


def factual_overall(f: dict[str, Any]) -> str:
    n = len(f["abnormal"])
    parts = [f"Not everything is normal: {n} of {f['n_channels']} channels {'is' if n == 1 else 'are'} abnormal."]
    parts += [f"{a['name']}: {_cond_text(a)}." for a in f["abnormal"]]
    parts += [f"{d['name']} has sent no rows" + (f" since {d['since']}." if d["since"] else " yet.")
              for d in f["feeds_down"]]
    return " ".join(parts)


def _feed_status(f: dict[str, Any]) -> str:
    if not f["feeds_down"]:
        return "All feeds are reporting."
    return " ".join(f"{d['name']} has sent no rows" + (f" since {d['since']}." if d["since"] else " yet.")
                    for d in f["feeds_down"])


def _feeds_in(text: str, feeds: list[str]) -> list[str]:
    low = text.lower().replace("flow rates", "flow_rates")
    return [x for x in feeds if re.search(rf"\b{re.escape(x.lower())}\b", low)]


def guard(summary: str, question: str, answer: str, feeds: list[str] | None = None) -> tuple[str, list[str]]:
    f = facts_from_summary(summary)
    low = answer.lower()
    fired: list[str] = []

    if f.get("no_data"):
        if re.match(r"\s*yes\b", low) or not re.search(r"no data|no feed has sent|no rows", low):
            return NO_DATA, ["no_data"]
        return answer, fired

    abnormal = f["abnormal"]
    if abnormal and _claims_all_clear(answer):
        return factual_overall(f), ["false_all_clear"]

    for word, cat in CATEGORIES.items():
        # A claim about the whole category ("the temperatures look normal", "all temperature
        # channels are normal"), not about named channels ("the magnet temperatures are normal").
        general = re.search(rf"(?:^|[.,;:!?]\s*|\b(?:the|all|all the|every|both)\s+){word}"
                            rf"(?:s|\s+channels|\s+readings|\s+sensors)?\b[^.]{{0,25}}?\b"
                            rf"(?:look|looks|are|is|seem|seems)\s+(?:all\s+)?"
                            rf"(?:normal|fine|ok|okay|good|healthy)", low)
        if general:
            missed = [a for a in abnormal if cat in a["name"].lower() and not _mentioned(a["name"], low)]
            if missed:
                text = (f"No. Abnormal {cat} channels: "
                        + "; ".join(f"{a['name']} ({_cond_text(a)})" for a in missed) + ".")
                return text, ["category"]

    known = feeds or sorted({d["name"] for d in f["feeds_down"]} | {"Flow_Rates", "Labjack", "QT"})
    down = {d["name"] for d in f["feeds_down"]}
    q_feeds = [x for x in _feeds_in(question, known) if x not in down and OUTAGE.search(question)]
    a_feeds = [x for s in _sentences(answer) if OUTAGE.search(s) and "not " not in s.lower()
               for x in _feeds_in(s, known) if x not in down]
    wrong = sorted(set(q_feeds + a_feeds))
    if wrong and f["feeds"]:
        return (f"{', '.join(wrong)} {'is' if len(wrong) == 1 else 'are'} not out. " + _feed_status(f)), ["feed_premise"]

    notes = []
    for s in f.get("stale", []):
        for name in s["channels"]:
            if _mentioned(name, low) and "normal" in low and "no current data" not in low \
                    and not re.search(rf"\b{re.escape(s['feed'].lower())}\b", low):
                notes.append(f"Note: {name} has no current data ({s['why']}).")
                fired.append("stale")
                break
    bad = [x for x in DECIMAL.findall(answer) if x.lstrip("-") not in summary and x not in question]
    if bad:
        notes.append(f"Note: {', '.join(sorted(set(bad)))} {'does' if len(set(bad)) == 1 else 'do'} not appear "
                     "in the summary.")
        fired.append("number")
    return (answer + (" " + " ".join(notes) if notes else "")), fired
