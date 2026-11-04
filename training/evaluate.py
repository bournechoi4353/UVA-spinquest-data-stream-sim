"""Score answers on the test set against each example's fact checks.

    # the deterministic template answerer (no GPU needed): a baseline
    python -m training.evaluate --data out/qa --answerer template

    # the base model without fine-tuning, and the fine-tuned adapter (GPU machine)
    python -m training.evaluate --data out/qa --answerer model
    python -m training.evaluate --data out/qa --answerer model --adapter out/model/adapter

An answer passes when it contains every ``must_include`` string, none of the
``must_not_include`` strings, and, for refusal examples (why, predictions,
unknown channels), says the summary does not cover it. A separate
hallucination check flags answers naming a channel or a clock time that is
not in the summary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REFUSAL = re.compile(r"(does not|doesn't|do not|don't|cannot|can't|can not|no information|not (say|mention|"
                     r"include|listed|in the summary)|unable to|isn't (in|listed))", re.I)
TIME = re.compile(r"\b\d{2}:\d{2}\b")
NAME = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[_.][A-Za-z0-9]+)+\b")


def check(answer: str, summary: str, question: str, checks: dict) -> dict:
    low = answer.lower()
    ok_inc = all(s.lower() in low for s in checks["must_include"] if not (checks["refusal"] and s in
                                                                            ("does not say", "can't predict",
                                                                             "does not mention")))
    ok_exc = not any(s.lower() in low for s in checks["must_not_include"])
    ok_ref = bool(REFUSAL.search(answer)) if checks["refusal"] else True
    times_bad = [t for t in TIME.findall(answer) if t not in summary]
    names_bad = [n for n in NAME.findall(answer) if n not in summary and n not in question]
    return {"pass": ok_inc and ok_exc and ok_ref, "hallucinated": bool(times_bad or names_bad),
            "bad": times_bad + names_bad}


def template_answers(rows):
    from training.qa import answer
    from training.summary_parse import facts_from_summary
    for r in rows:
        summary = r["messages"][1]["content"].split("Summary:\n", 1)[1].split("\n\nQuestion: ")[0]
        question = r["messages"][1]["content"].split("\n\nQuestion: ", 1)[1]
        yield answer(facts_from_summary(summary), question)


def model_answers(rows, model_name: str, adapter: str | None, batch_size: int, max_new: int):
    import torch
    from transformers import AutoTokenizer

    from training.modeling import chat_prompt, load_model
    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(model_name, torch.bfloat16, "cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        prompts = [chat_prompt(tok, r["messages"][:-1], add_generation_prompt=True, tokenize=False)
                   for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for seq in out:
            yield tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\r  generated {min(i + batch_size, len(rows))}/{len(rows)}", end="", flush=True)
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="out/qa")
    p.add_argument("--split", default="test")
    p.add_argument("--answerer", choices=["template", "model"], default="template")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter")
    p.add_argument("--limit", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--out", help="write per-example results as JSONL")
    args = p.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.data, f"{args.split}.jsonl").read_text(encoding="utf-8")
            .splitlines() if line.strip()][:args.limit]
    answers = (template_answers(rows) if args.answerer == "template"
               else model_answers(rows, args.model, args.adapter, args.batch_size, args.max_new_tokens))
    by_intent = defaultdict(lambda: [0, 0, 0])
    results = []
    for r, a in zip(rows, answers):
        summary = r["messages"][1]["content"]
        question = summary.split("\n\nQuestion: ", 1)[1]
        res = check(a, summary, question, r["checks"])
        d = by_intent[r["intent"]]
        d[0] += 1
        d[1] += res["pass"]
        d[2] += res["hallucinated"]
        results.append({"intent": r["intent"], "question": question, "answer": a,
                        "reference": r["messages"][2]["content"], **res})
    label = args.answerer if args.answerer == "template" else (args.adapter or args.model)
    print(f"Answerer: {label}, split {args.split}, {len(results)} examples")
    print(f"{'intent':18} {'n':>5} {'pass':>7} {'hallucinated':>13}")
    for k in sorted(by_intent):
        n, ok, hal = by_intent[k]
        print(f"{k:18} {n:>5} {ok / n:>7.1%} {hal / n:>13.1%}")
    n = len(results)
    print(f"{'ALL':18} {n:>5} {sum(r['pass'] for r in results) / n:>7.1%} "
          f"{sum(r['hallucinated'] for r in results) / n:>13.1%}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
