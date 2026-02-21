#!/usr/bin/env python3
"""RL-3 Environment Tool Use — adapter for rl_core.

Reward: execution-based grading (generated code passes test cases = 1.0).
Run:  python demos/rl/adapters/environment_tooluse.py
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

FEWSHOT = """Q: Write a function `double(x)` that returns x * 2.
A: ```python
def double(x):
    return x * 2
```

"""

PROBLEMS = [
    {"q": "Write `add(a, b)` that returns a + b.", "tests": [("add(1,2)", 3), ("add(-1,1)", 0)]},
    {"q": "Write `square(x)` that returns x squared.", "tests": [("square(3)", 9), ("square(-2)", 4)]},
    {"q": "Write `max2(a, b)` that returns the larger.", "tests": [("max2(3,5)", 5), ("max2(7,2)", 7)]},
    {"q": "Write `is_even(n)` returning True if even.", "tests": [("is_even(4)", True), ("is_even(7)", False)]},
    {"q": "Write `abs_val(x)` returning absolute value.", "tests": [("abs_val(-5)", 5), ("abs_val(3)", 3)]},
]


def _extract_code(response: str) -> str | None:
    match = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
    if match:
        return match[-1].strip()
    if "def " in response:
        return response[response.find("def "):].strip()
    return None


class EnvironmentToolUseAdapter(RLAdapter):

    def build_dataset(self) -> list[dict]:
        return [random.choice(PROBLEMS) for _ in range(50)]

    def make_prompt(self, sample: dict, tokenizer: Any) -> list[int]:
        return tokenizer.encode(FEWSHOT + f"Q: {sample['q']}\nA:")

    def compute_reward(self, response: str, sample: dict) -> float:
        code = _extract_code(response)
        if not code:
            return 0.0
        try:
            ns: dict[str, Any] = {}
            exec(code, ns)  # noqa: S102
            for expr, expected in sample["tests"]:
                if eval(expr, ns) != expected:  # noqa: S307
                    return 0.0
            return 1.0
        except Exception:
            return 0.0

    def evaluate(self, step: int, rewards: list[float], num_datums: int) -> None:
        accuracy = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        print(f"Step {step}: accuracy={accuracy:.1%}, datums={num_datums}")


if __name__ == "__main__":
    cfg = RLConfig.from_env()
    cfg.steps = int(os.environ.get("MINT_RL_STEPS", "10"))
    cfg.batch = int(os.environ.get("MINT_RL_BATCH", "8"))
    cfg.max_tokens = int(os.environ.get("MINT_RL_MAX_TOKENS", "256"))
    run_grpo(EnvironmentToolUseAdapter(), cfg)
