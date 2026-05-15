#!/usr/bin/env python3
"""MinT Recipe: RL Hyperparameter Sweep

Run a real RL sweep on a tiny arithmetic MessageEnv using recipe.rl.train.main().
The default grid is 2 KL values x 2 temperatures x 2 group sizes = 8 configs.

Run:
  MINT_API_KEY=sk-xxx python recipes/rl_hyperparameters.py

Useful overrides:
  MINT_RL_STEPS=1 MINT_RL_KL_COEFS=0.0,0.02 MINT_RL_TEMPS=0.7,1.0 MINT_RL_GROUPS=2,4 python recipes/rl_hyperparameters.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
RL_STEPS = int(os.environ.get("MINT_RL_STEPS", "1"))
BATCH_SIZE = int(os.environ.get("MINT_RL_BATCH", "1"))
MAX_TOKENS = int(os.environ.get("MINT_RL_MAX_TOKENS", "64"))
LOG_ROOT = Path(os.environ.get("MINT_LOG_ROOT", "/tmp"))


PROBLEMS = [
    ("What is 2 + 3? Answer with only the number.", "5"),
    ("What is 4 * 6? Answer with only the number.", "24"),
    ("What is 12 - 7? Answer with only the number.", "5"),
    ("What is 18 / 3? Answer with only the number.", "6"),
]


def _parse_float_list(env_name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_int_list(env_name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env_name)
    if not raw:
        return default
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _first_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


class ArithmeticMessageEnv(MessageEnv):
    """Single-turn arithmetic environment for quick RL sweeps."""

    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    async def initial_observation(self) -> list[dict]:
        return [
            {
                "role": "system",
                "content": "You are a precise calculator. Reply with only the final number.",
            },
            {"role": "user", "content": self.question},
        ]

    async def step(self, message: dict) -> MessageStepResult:
        content = extract_content(message).strip()
        prediction = _first_number(content)
        correct = prediction == self.answer
        reward = 1.0 if correct else -0.25
        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=[],
            metrics={
                "correct": float(correct),
                "invalid_format": float(prediction is None),
            },
        )


@dataclass(frozen=True)
class ArithmeticEnvGroupBuilder(recipe.rl.types.EnvGroupBuilder):
    question: str
    answer: str
    group_size: int
    renderer_name: str
    model_name: str

    async def make_envs(self) -> Sequence[recipe.rl.types.Env]:
        tokenizer = get_tokenizer(self.model_name)
        renderer = recipe.renderers.get_renderer(self.renderer_name, tokenizer)
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=ArithmeticMessageEnv(self.question, self.answer),
                max_trajectory_tokens=512,
                max_generation_tokens=MAX_TOKENS,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["arithmetic"]


class ArithmeticRLDataset(recipe.rl.types.RLDataset):
    def __init__(self, problems, batch_size, group_size, renderer_name, model_name):
        self.problems = problems
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.model_name = model_name

    def __len__(self) -> int:
        return max(1, (len(self.problems) + self.batch_size - 1) // self.batch_size)

    def get_batch(self, index: int):
        start = (index * self.batch_size) % len(self.problems)
        batch = self.problems[start : start + self.batch_size]
        if len(batch) < self.batch_size:
            batch += self.problems[: self.batch_size - len(batch)]
        return [
            ArithmeticEnvGroupBuilder(
                question=question,
                answer=answer,
                group_size=self.group_size,
                renderer_name=self.renderer_name,
                model_name=self.model_name,
            )
            for question, answer in batch
        ]


@chz.chz
class ArithmeticRLDatasetBuilder(recipe.rl.types.RLDatasetBuilder):
    batch_size: int
    group_size: int
    renderer_name: str
    model_name: str

    async def __call__(self):
        return (
            ArithmeticRLDataset(
                PROBLEMS,
                self.batch_size,
                self.group_size,
                self.renderer_name,
                self.model_name,
            ),
            None,
        )


def _read_last_metrics(log_path: Path) -> dict[str, Any]:
    metrics_path = log_path / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    last = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            last = json.loads(line)
        except json.JSONDecodeError:
            continue
    return last


async def run_config(
    *,
    renderer_name: str,
    kl_coef: float,
    temperature: float,
    group_size: int,
    run_id: str,
) -> dict[str, Any]:
    label = f"kl={kl_coef}, temp={temperature}, group={group_size}"
    safe_kl = str(kl_coef).replace(".", "p")
    safe_temp = str(temperature).replace(".", "p")
    log_path = LOG_ROOT / f"mint-rl-sweep-{run_id}-kl{safe_kl}-t{safe_temp}-g{group_size}"

    print(f"\n--- Training config: {label} ---")
    print(f"Log path: {log_path}")

    kl_reference_config = (
        recipe.rl.train.KLReferenceConfig(base_model=MODEL) if kl_coef > 0 else None
    )

    config = recipe.rl.train.Config(
        learning_rate=float(os.environ.get("MINT_RL_LR", "1e-5")),
        dataset_builder=ArithmeticRLDatasetBuilder(
            batch_size=BATCH_SIZE,
            group_size=group_size,
            renderer_name=renderer_name,
            model_name=MODEL,
        ),
        model_name=MODEL,
        renderer_name=renderer_name,
        lora_rank=RANK,
        max_tokens=MAX_TOKENS,
        temperature=temperature,
        kl_penalty_coef=kl_coef,
        kl_reference_config=kl_reference_config,
        loss_fn="importance_sampling",
        log_path=str(log_path),
        max_steps=RL_STEPS,
        save_every=999,
        eval_every=999,
        ttl_seconds=3600,
        num_groups_to_log=1,
        rollout_json_export=False,
    )
    await recipe.rl.train.main(config=config)
    metrics = _read_last_metrics(log_path)
    mean_reward = None
    correct_rate = None
    for key, value in metrics.items():
        if key in {"env/all/reward/total", "env/all/mean_episode_reward"} and isinstance(value, int | float):
            mean_reward = float(value)
        if key.endswith("/correct") and isinstance(value, int | float):
            correct_rate = float(value)
    return {
        "kl_penalty_coef": kl_coef,
        "temperature": temperature,
        "group_size": group_size,
        "steps": RL_STEPS,
        "mean_episode_reward": mean_reward,
        "correct_rate": correct_rate,
        "log_path": str(log_path),
    }


async def run_sweep() -> list[dict[str, Any]]:
    kl_coefs = _parse_float_list("MINT_RL_KL_COEFS", [0.0, 0.02])
    temperatures = _parse_float_list("MINT_RL_TEMPS", [0.7, 1.0])
    group_sizes = _parse_int_list("MINT_RL_GROUPS", [2, 4])
    renderer_name = recipe.get_recommended_renderer_name(MODEL)
    run_id = str(int(time.time()))

    print("\n=== RL Hyperparameter Sweep ===")
    print(f"Model:          {MODEL}")
    print(f"Renderer:       {renderer_name}")
    print(f"Steps/config:   {RL_STEPS}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"KL coefficients:{kl_coefs}")
    print(f"Temperatures:   {temperatures}")
    print(f"Group sizes:    {group_sizes}")
    print(f"Grid size:      {len(kl_coefs)} x {len(temperatures)} x {len(group_sizes)} = {len(kl_coefs) * len(temperatures) * len(group_sizes)} configs")

    results: list[dict[str, Any]] = []
    for kl_coef in kl_coefs:
        for temperature in temperatures:
            for group_size in group_sizes:
                results.append(
                    await run_config(
                        renderer_name=renderer_name,
                        kl_coef=kl_coef,
                        temperature=temperature,
                        group_size=group_size,
                        run_id=run_id,
                    )
                )
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== RL Grid Summary ===")
    print(f"{'KL':<8} {'Temp':<8} {'Group':<8} {'Steps':<8} {'Mean reward':<14} {'Correct':<10} {'Log path'}")
    print("-" * 110)
    for result in results:
        reward = result["mean_episode_reward"]
        reward_text = f"{reward:.3f}" if isinstance(reward, float) else "n/a"
        correct = result["correct_rate"]
        correct_text = f"{correct:.2f}" if isinstance(correct, float) else "n/a"
        print(
            f"{result['kl_penalty_coef']:<8.2f} "
            f"{result['temperature']:<8.1f} "
            f"{result['group_size']:<8} "
            f"{result['steps']:<8} "
            f"{reward_text:<14} "
            f"{correct_text:<10} "
            f"{result['log_path']}"
        )


def main() -> int:
    try:
        print("Connecting to MinT server...")
        print(f"Endpoint: {configured_base_url()}")
        _service_client, capabilities = make_service_client()
        supported = supported_model_count(capabilities)
        if supported is None:
            print("Auth preflight: OK")
        else:
            print(f"Auth preflight: OK ({supported} supported models)")

        results = asyncio.run(run_sweep())
        print_summary(results)
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
