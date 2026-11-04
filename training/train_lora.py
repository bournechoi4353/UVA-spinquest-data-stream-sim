"""LoRA fine-tune of a small instruct model on the Q&A dataset.

Run on the GPU machine, inside its own fresh virtual environment (never the
machine's main Python or conda environment):

    python3 -m venv .venv-train
    .venv-train/bin/pip install -r requirements-train.txt
    .venv-train/bin/python -m training.train_lora --data out/qa --out out/model

Defaults: Qwen/Qwen3.5-4B, LoRA rank 16 on every linear layer of the language
model (not the vision tower), bf16, 2 epochs, learning rate 2e-4, loss on the assistant answer
only (prompt tokens are masked). The adapter and tokenizer are saved to --out.

Only the summary and the question go into the prompt, exactly as the bot will
send them (see training/qa.py: to_chat).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def encode(tok, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Token ids and labels; labels are -100 everywhere except the assistant answer."""
    from training.modeling import as_ids, chat_prompt
    prompt_ids = as_ids(chat_prompt(tok, messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = as_ids(chat_prompt(tok, messages, add_generation_prompt=False, tokenize=True))
    if full_ids[:len(prompt_ids)] != prompt_ids:     # fall back to plain concatenation
        answer_ids = tok(messages[-1]["content"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    return full_ids[:max_len], labels[:max_len]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="out/qa")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--out", default="out/model")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--limit", type=int, help="use only the first N training examples (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    from training.modeling import load_model, lora_targets

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(args.model, dtype, device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    targets = lora_targets(model)
    print(f"LoRA targets: {targets}")
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=targets, exclude_modules=r".*(visual|vision).*"))
    model.print_trainable_parameters()

    data = Path(args.data)
    train = [encode(tok, r["messages"], args.max_len) for r in load_rows(data / "train.jsonl", args.limit)]
    val = [encode(tok, r["messages"], args.max_len) for r in load_rows(data / "val.jsonl", 400)]
    print(f"train {len(train)} examples, val {len(val)}; longest {max(len(x[0]) for x in train)} tokens")

    def batches(items, bs, shuffle):
        idx = list(range(len(items)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), bs):
            chunk = [items[j] for j in idx[i:i + bs]]
            n = max(len(c[0]) for c in chunk)
            ids = torch.full((len(chunk), n), tok.pad_token_id)
            lab = torch.full((len(chunk), n), -100)
            att = torch.zeros((len(chunk), n), dtype=torch.long)
            for k, (a, b) in enumerate(chunk):
                ids[k, :len(a)] = torch.tensor(a)
                lab[k, :len(b)] = torch.tensor(b)
                att[k, :len(a)] = 1
            yield ids.to(device), lab.to(device), att.to(device)

    steps_per_epoch = math.ceil(len(train) / args.batch_size / args.grad_accum)
    total = int(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, total // 20), total)

    def evaluate() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for ids, lab, att in batches(val, args.batch_size, False):
                losses.append(model(input_ids=ids, attention_mask=att, labels=lab).loss.item())
        model.train()
        return sum(losses) / max(len(losses), 1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    model.train()
    step, t0 = 0, time.time()
    print(f"val loss before training: {evaluate():.4f}")
    epoch = 0
    while step < total:
        for k, (ids, lab, att) in enumerate(batches(train, args.batch_size, True)):
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / args.grad_accum
            loss.backward()
            if (k + 1) % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0 or step == total:
                rec = {"step": step, "epoch": epoch, "loss": loss.item() * args.grad_accum,
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
            if step >= total:
                break
        epoch += 1
        v = evaluate()
        print(f"epoch {epoch}: val loss {v:.4f}", flush=True)
        log.write(json.dumps({"epoch": epoch, "val_loss": v}) + "\n")
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"saved adapter to {out / 'adapter'} ({round(time.time() - t0)} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
