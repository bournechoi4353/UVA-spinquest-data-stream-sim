"""Model loading helpers shared by training, evaluation and the bot.

The default model is Qwen/Qwen3.5-4B. It is published as a vision-language
model (Qwen3_5ForConditionalGeneration), with hybrid linear-attention layers.
We use it for text only:

* ``load_model`` tries the text-only causal LM class first and falls back to
  the image-text-to-text class (both accept text-only input).
* ``lora_targets`` lists every linear layer of the language model, skipping the
  vision tower and the output head, so LoRA works whatever the layer names are.
* ``chat_prompt`` applies the chat template with thinking turned off
  (Qwen3.5 otherwise starts answers with a reasoning block).
"""

from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_SKIP = ("visual", "vision", "lm_head", "embed", "merger", "patch")


def load_model(name: str, dtype, device: str):
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    return model.to(device)


def lora_targets(model) -> list[str]:
    """Names of the language model's linear layers (leaf names, deduplicated)."""
    import torch

    names = set()
    for full, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(s in full.lower() for s in _SKIP):
            names.add(full.split(".")[-1])
    return sorted(names)


def chat_prompt(tok, messages: list[dict], add_generation_prompt: bool, tokenize: bool):
    """Chat template with thinking disabled where the template supports it."""
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt,
                                       tokenize=tokenize, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=tokenize)


def as_ids(x) -> list[int]:
    """apply_chat_template(tokenize=True) returns a list or a BatchEncoding depending on version."""
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)
