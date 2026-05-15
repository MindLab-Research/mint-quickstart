#!/usr/bin/env python3
"""MinT Recipe: Multi-Turn RL

Real multi-turn RL using mint.recipe. The model interacts with a Calculator
environment over multiple turns, and is trained via GRPO
(importance_sampling loss) so it learns to use calc() and give correct answers.

Run:
  MINT_API_KEY=sk-xxx python recipes/multi_turn_rl.py

All training runs against a remote MinT server.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from _common import (  # noqa: E402
    configured_base_url,
    extract_content,
    supported_model_count,
    make_service_client,
)

import chz  # noqa: E402
import mint.recipe as recipe  # noqa: E402
from mint.recipe import EnvFromMessageEnv, MessageEnv, MessageStepResult, get_tokenizer  # noqa: E402


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
RL_STEPS = int(os.environ.get("MINT_RL_STEPS", "3"))
GROUP_SIZE = int(os.environ.get("MINT_GROUP_SIZE", "4"))


# ---------------------------------------------------------------------------
# Environment: Calculator with tool use
# ---------------------------------------------------------------------------

def _safe_calc(expr: str) -> str:
    """Evaluate a small arithmetic expression with an AST allowlist."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expr):
        return "Error: only arithmetic operators allowed"
    try:
        import ast
        import math

        def calc_node(node):
            if isinstance(node, ast.Expression):
                return calc_node(node.body)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.UnaryOp):
                value = calc_node(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return value
                if isinstance(node.op, ast.USub):
                    return -value
            if isinstance(node, ast.BinOp):
                left = calc_node(node.left)
                right = calc_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if abs(right) > 8:
                        raise ValueError("exponent too large")
                    return left ** right
            raise ValueError(f"unsupported operation {type(node).__name__}")

        result = calc_node(ast.parse(expr, mode="eval"))
        if not isinstance(result, (int, float)) or not math.isfinite(result):
            return "Error: invalid expression"
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except Exception:
        return "Error: invalid expression"


class CalculatorEnv(MessageEnv):
    """Model must use calc(expr) to solve math. Demonstrates multi-turn tool use."""

    def __init__(self, question: str, answer: float):
        self.question = question
        self.answer = answer
        self.turns = 0
        self.max_turns = 3

    async def initial_observation(self) -> list[dict]:
        return [
            {"role": "system", "content": (
                "You can use calc(expr) to evaluate math expressions. "
                "Give your final answer as: Answer: <number>"
            )},
            {"role": "user", "content": self.question},
        ]

    async def step(self, message: dict) -> MessageStepResult:
        content = extract_content(message)
        self.turns += 1

        answer_match = re.search(r"Answer:\s*([\d.]+)", content)
        if answer_match:
            correct = abs(float(answer_match.group(1)) - self.answer) < 0.01
            return MessageStepResult(
                reward=1.0 if correct else -0.5,
                episode_done=True,
                next_messages=[],
                metrics={"correct": float(correct), "turns": self.turns},
            )

        calc_match = re.search(r"calc\((.+?)\)", content)
        if calc_match:
            result = _safe_calc(calc_match.group(1))
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=[{"role": "user", "content": f"Result: {result}"}],
                metrics={},
            )

        if self.turns >= self.max_turns:
            return MessageStepResult(
                reward=-1.0, episode_done=True, next_messages=[],
                metrics={"timeout": 1.0},
            )

        return MessageStepResult(
            reward=0.0, episode_done=True, next_messages=[],
            metrics={"invalid_format": 1.0},
        )


# ---------------------------------------------------------------------------
# Dataset wiring: EnvGroupBuilder + RLDataset + RLDatasetBuilder
# ---------------------------------------------------------------------------

PROBLEMS = [
    ("What is 12 * 34?", 408.0),
    ("What is 99 + 101?", 200.0),
    ("What is 256 / 8?", 32.0),
    ("What is 7 * 13?", 91.0),
]


@dataclass(frozen=True)
class CalculatorEnvGroupBuilder(recipe.rl.types.EnvGroupBuilder):
    question: str
    answer: float
    group_size: int
    renderer_name: str
    model_name: str

    async def make_envs(self) -> Sequence[recipe.rl.types.Env]:
        tok = get_tokenizer(self.model_name)
        rend = recipe.renderers.get_renderer(self.renderer_name, tok)
        return [
            EnvFromMessageEnv(
                renderer=rend,
                message_env=CalculatorEnv(self.question, self.answer),
                max_trajectory_tokens=2048,
                max_generation_tokens=256,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["calculator"]


class CalculatorDataset(recipe.rl.types.RLDataset):
    def __init__(self, problems, batch_size, group_size, renderer_name, model_name):
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.model_name = model_name

    def __len__(self):
        return max(1, len(self.problems) // self.batch_size)

    def get_batch(self, batch_idx):
        start = (batch_idx * self.batch_size) % len(self.problems)
        batch = self.problems[start : start + self.batch_size]
        return [
            CalculatorEnvGroupBuilder(
                q, a, self.group_size, self.renderer_name, self.model_name
            )
            for q, a in batch
        ]


@chz.chz
class CalculatorDatasetBuilder(recipe.rl.types.RLDatasetBuilder):
    batch_size: int = 2
    group_size: int = 4
    renderer_name: str = ""
    model_name: str = ""

    async def __call__(self):
        return CalculatorDataset(
            PROBLEMS, self.batch_size, self.group_size,
            self.renderer_name, self.model_name,
        ), None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run multi-turn RL training with a Calculator environment."""
    try:
        base_url = configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        _service_client, capabilities = make_service_client()
        print(f"Server supports {supported_model_count(capabilities)} models")

        renderer_name = recipe.get_recommended_renderer_name(MODEL)

        print(f"\nModel:      {MODEL}")
        print(f"LoRA Rank:  {RANK}")
        print(f"RL Steps:   {RL_STEPS}")
        print(f"Group Size: {GROUP_SIZE}")
        print(f"Renderer:   {renderer_name}")
        print(f"Problems:   {len(PROBLEMS)}")

        config = recipe.rl.train.Config(
            learning_rate=1e-5,
            dataset_builder=CalculatorDatasetBuilder(
                batch_size=2,
                group_size=GROUP_SIZE,
                renderer_name=renderer_name,
                model_name=MODEL,
            ),
            model_name=MODEL,
            renderer_name=renderer_name,
            lora_rank=RANK,
            max_tokens=256,
            temperature=0.8,
            kl_penalty_coef=0.0,
            loss_fn="importance_sampling",
            log_path="/tmp/mint-multiturn-rl",
            max_steps=RL_STEPS,
            save_every=999,
            eval_every=999,
        )

        print("\nStarting RL training loop...")
        asyncio.run(recipe.rl.train.main(config=config))
        print("\nTraining complete.")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
