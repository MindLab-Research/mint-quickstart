#!/usr/bin/env python3
"""RL-2 Preference Chat — adapter for rl_core.

Reward: proxy helpfulness score (length + structure + diversity).
Run:  python demos/rl/adapters/preference_chat.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_core import RLAdapter, RLConfig, run_grpo  # noqa: E402

PROMPTS = [
    "Explain what a variable is in programming.",
    "Write a short poem about the ocean.",
    "What are three benefits of exercise?",
    "Describe how to make a cup of tea.",
    "Why is the sky blue?",
    "Give tips for better sleep.",
    "What is machine learning?",
    "How do plants make food?",
]


class PreferenceChatAdapter(RLAdapter):

    def build_dataset(self) -> list[str]:
        return [random.choice(PROMPTS) for _ in range(50)]

    def make_prompt(self, sample: str, tokenizer: Any) -> list[int]:
        messages = [{"role": "user", "content": sample}]
        if hasattr(tokenizer, "apply_chat_template"):
            return list(tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            ))
        return tokenizer.encode(f"User: {sample}\nAssistant:")

    def compute_reward(self, response: str, sample: str) -> float:
        r = 0.0
        words = len(response.split())
        if 20 <= words <= 100:
            r += 0.4
        elif 10 <= words < 20 or 100 < words <= 150:
            r += 0.2
        if response.count(".") >= 2:
            r += 0.3
        unique_words = len(set(response.lower().split()))
        if words > 0 and unique_words > words * 0.5:
            r += 0.3
        return min(r, 1.0)

    def evaluate(self, step: int, rewards: list[float], num_datums: int) -> None:
        avg = sum(rewards) / len(rewards) if rewards else 0.0
        print(f"Step {step}: avg_reward={avg:.3f}, datums={num_datums}")


if __name__ == "__main__":
    cfg = RLConfig.from_env()
    cfg.steps = int(os.environ.get("MINT_RL_STEPS", "10"))
    cfg.batch = int(os.environ.get("MINT_RL_BATCH", "8"))
    cfg.max_tokens = int(os.environ.get("MINT_RL_MAX_TOKENS", "128"))
    run_grpo(PreferenceChatAdapter(), cfg)
