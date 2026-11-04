"""Discord bot: answers questions about the monitored system from the latest summary.

The bot never sees raw data. It reads ``summary.txt`` written by
``scripts/run_monitor.py --follow`` and passes the summary and the question to
the answerer (the fine-tuned model, or the template answerer as a fallback).

Setup (in its own virtual environment):
    pip install -r requirements-bot.txt            # plus requirements-train.txt for the model
    export DISCORD_BOT_TOKEN=...                   # never put the token in a file in the repo
    python -m bot.discord_bot --monitor-dir out/monitor/ptdb --adapter out/model/adapter

In Discord: mention the bot with a question, or use "!ask <question>" or "!status".
In the Discord developer portal, enable the "Message Content" intent for the bot.

Offline check without Discord:
    python -m bot.discord_bot --monitor-dir out/monitor/seed1_stress --ask "anything wrong?"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.answerer import ModelAnswerer, TemplateAnswerer, read_summary  # noqa: E402

MAX_DISCORD = 1900


def make_answerer(args):
    if args.adapter or args.model_only:
        return ModelAnswerer(args.base_model, args.adapter)
    return TemplateAnswerer()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--monitor-dir", required=True, help="folder with summary.txt from run_monitor.py")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--adapter", help="LoRA adapter folder from training/train_lora.py")
    p.add_argument("--model-only", action="store_true", help="use the base model without an adapter")
    p.add_argument("--ask", help="answer one question offline and exit (no Discord)")
    args = p.parse_args(argv)

    answerer = make_answerer(args)
    if args.ask:
        summary = read_summary(args.monitor_dir)
        print(summary, "\n")
        print(f"[{answerer.name}] {answerer(summary, args.ask)}")
        return 0

    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        print("Set DISCORD_BOT_TOKEN in the environment (never in a file in the repo).", file=sys.stderr)
        return 2
    import discord

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(f"Logged in as {client.user} (answerer: {answerer.name})")

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        text = message.content.strip()
        mentioned = client.user in message.mentions
        if text.startswith("!status"):
            await message.channel.send(read_summary(args.monitor_dir)[:MAX_DISCORD])
            return
        if text.startswith("!ask "):
            question = text[5:].strip()
        elif mentioned:
            question = text.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()
        else:
            return
        if not question:
            return
        summary = read_summary(args.monitor_dir)
        async with message.channel.typing():
            reply = answerer(summary, question)
        await message.channel.send(reply[:MAX_DISCORD])

    client.run(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
