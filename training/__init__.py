"""Training data and fine-tuning for the small answer-phrasing LLM.

    qa.py            facts from a state-store snapshot, and question/answer generation
    make_dataset.py  snapshots from simulated runs -> train/val/test JSONL (chat format)
    train_lora.py    LoRA fine-tune of a small instruct model (run on the GPU machine)
    evaluate.py      fact-level scoring of model answers against each example's checks

The model only ever sees the summary text and the question. Answers are
grounded in the summary: no causes, no predictions, no channels that are not
in the summary.
"""
