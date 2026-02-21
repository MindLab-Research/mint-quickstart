#!/usr/bin/env python3
"""RL-1 Verifiable Math — adapter for rl_core.

Reward: exact-match integer grading (1.0 correct, 0.0 wrong).
Run:  python demos/rl/adapters/verifiable_math.py
"""

from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_core import RLAdapter, RLConfig, run_grpo  # noqa: E402


class VerifiableMathAdapter(RLAdapter):
    FEWSHOT = "Q: What is 4 + 5?\nA: 9\n\n"

    def build_dataset(self) -> list[tuple[str, int]]:
        return [
            (f"Q: What is {random.randint(0, 99)} + {random.randint(0, 99)}?\nA:", a + b)
            for a, b in [(random.randint(0, 99), random.randint(0, 99)) for _ in range(50)]
        ]

    def make_prompt(self, sample: tuple[str, int], tokenizer: Any) -> list[int]:
        question, _ = sample
        return tokenizer.encode(self.FEWSHOT + question)

    def compute_reward(self, response: str, sample: tuple[str, int]) -> float:
        _, answer = sample
        match = re.search(r"-?\d+", response)
        return 1.0 if match and int(match.group()) == answer else 0.0

    def evaluate(self, step: int, rewards: list[float], num_datums: int) -> None:
        accuracy = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        print(f"Step {step}: accuracy={accuracy:.1%}, datums={num_datums}")


if __name__ == "__main__":
    cfg = RLConfig.from_env()
    cfg.steps = int(os.environ.get("MINT_RL_STEPS", "5"))
    cfg.batch = int(os.environ.get("MINT_RL_BATCH", "10"))
    cfg.max_tokens = int(os.environ.get("MINT_RL_MAX_TOKENS", "8"))
    run_grpo(VerifiableMathAdapter(), cfg)
