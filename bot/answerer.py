"""Turn a question into an answer, from the latest summary only.

Two answerers with the same interface:

* TemplateAnswerer: the deterministic keyword answerer (no GPU, no model).
* ModelAnswerer: the fine-tuned LLM (base model plus LoRA adapter).

Both receive only the summary text and the question, exactly as in training.
"""

from __future__ import annotations

from pathlib import Path

from training.qa import answer as template_answer
from training.qa import to_chat
from training.summary_parse import facts_from_summary


def read_summary(monitor_dir: str | Path) -> str:
    path = Path(monitor_dir) / "summary.txt"
    if not path.is_file():
        return "Status: no data yet."
    return path.read_text(encoding="utf-8").strip()


class TemplateAnswerer:
    name = "template"

    def __call__(self, summary: str, question: str) -> str:
        return template_answer(facts_from_summary(summary), question)


class ModelAnswerer:
    name = "model"

    def __init__(self, base_model: str, adapter: str | None = None, max_new_tokens: int = 200):
        import torch
        from transformers import AutoTokenizer

        from training.modeling import load_model

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(adapter or base_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = load_model(base_model, dtype, device)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    def __call__(self, summary: str, question: str) -> str:
        from training.modeling import chat_prompt
        prompt = chat_prompt(self.tok, to_chat(summary, question), add_generation_prompt=True, tokenize=False)
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
