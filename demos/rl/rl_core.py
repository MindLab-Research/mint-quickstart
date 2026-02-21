"""Shared GRPO training loop for all RL demos.

Owns the generic RL loop: sample -> reward -> advantage -> train -> log.
Each adapter provides task-specific logic via the RLAdapter protocol.

All training runs against a remote MinT server.
This module does NOT start any backend services locally.
"""

from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MINT_SRC = REPO_ROOT / "mindlab-toolkit" / "src"
if MINT_SRC.exists() and str(MINT_SRC) not in sys.path:
    sys.path.insert(0, str(MINT_SRC))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(REPO_ROOT / ".env")

import mint  # noqa: E402
from mint import types  # noqa: E402


@dataclass
class RLConfig:
    model: str = "Qwen/Qwen3-0.6B"
    rank: int = 16
    steps: int = 5
    batch: int = 8
    group: int = 4
    lr: float = 1e-4
    max_tokens: int = 16
    temperature: float = 1.0

    @classmethod
    def from_env(cls) -> RLConfig:
        return cls(
            model=os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B"),
            rank=int(os.environ.get("MINT_LORA_RANK", "16")),
            steps=int(os.environ.get("MINT_RL_STEPS", "5")),
            batch=int(os.environ.get("MINT_RL_BATCH", "8")),
            group=int(os.environ.get("MINT_RL_GROUP", "4")),
            lr=float(os.environ.get("MINT_RL_LR", "1e-4")),
            max_tokens=int(os.environ.get("MINT_RL_MAX_TOKENS", "16")),
            temperature=float(os.environ.get("MINT_RL_TEMPERATURE", "1.0")),
        )


class RLAdapter(ABC):
    """Each demo implements this to plug into the shared loop."""

    @abstractmethod
    def build_dataset(self) -> list[Any]:
        """Return a list of task items for one training step."""

    @abstractmethod
    def make_prompt(self, sample: Any, tokenizer: Any) -> list[int]:
        """Convert a dataset sample into prompt token ids."""

    @abstractmethod
    def compute_reward(self, response: str, sample: Any) -> float:
        """Score a single generated response. Returns scalar reward."""

    def evaluate(self, step: int, rewards: list[float], num_datums: int) -> None:
        """Log metrics. Default prints accuracy."""
        accuracy = sum(1 for r in rewards if r > 0) / len(rewards) if rewards else 0.0
        print(f"Step {step}: accuracy={accuracy:.1%}, datums={num_datums}")


def run_grpo(adapter: RLAdapter, cfg: RLConfig | None = None) -> str:
    """Execute the shared GRPO loop. Returns final checkpoint path."""
    if cfg is None:
        cfg = RLConfig.from_env()

    service_client = mint.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=cfg.model,
        rank=cfg.rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    tokenizer = training_client.get_tokenizer()
    print(f"Model: {cfg.model}, Vocab: {tokenizer.vocab_size:,}")

    for step in range(1, cfg.steps + 1):
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"rl-step-{step}"
        )

        dataset = adapter.build_dataset()
        training_datums: list[types.Datum] = []
        all_rewards: list[float] = []

        for sample in dataset[: cfg.batch]:
            prompt_tokens = adapter.make_prompt(sample, tokenizer)

            result = sampling_client.sample(
                prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                num_samples=cfg.group,
                sampling_params=types.SamplingParams(
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                    stop_token_ids=[tokenizer.eos_token_id],
                ),
            ).result()

            group_rewards: list[float] = []
            group_responses: list[list[int]] = []
            group_logprobs: list[list[float]] = []

            for seq in result.sequences:
                response_text = tokenizer.decode(seq.tokens)
                reward = adapter.compute_reward(response_text, sample)
                group_rewards.append(reward)
                group_responses.append(list(seq.tokens))
                group_logprobs.append(
                    list(seq.logprobs or [0.0] * len(seq.tokens))
                )

            all_rewards.extend(group_rewards)
            mean_reward = sum(group_rewards) / len(group_rewards)
            advantages = [r - mean_reward for r in group_rewards]
            if all(a == 0.0 for a in advantages):
                continue

            prefix_len = len(prompt_tokens) - 1
            for resp_tokens, logprobs, adv in zip(
                group_responses, group_logprobs, advantages
            ):
                if not resp_tokens:
                    continue
                full_tokens = prompt_tokens + resp_tokens
                r_len = len(resp_tokens)
                training_datums.append(
                    types.Datum(
                        model_input=types.ModelInput.from_ints(
                            tokens=full_tokens[:-1]
                        ),
                        loss_fn_inputs={
                            "target_tokens": full_tokens[1:],
                            "weights": [0.0] * prefix_len + [1.0] * r_len,
                            "logprobs": [0.0] * prefix_len + logprobs,
                            "advantages": [0.0] * prefix_len + [adv] * r_len,
                        },
                    )
                )

        if training_datums:
            training_client.forward_backward(
                training_datums, loss_fn="importance_sampling"
            ).result()
            training_client.optim_step(
                types.AdamParams(learning_rate=cfg.lr)
            ).result()

        adapter.evaluate(step, all_rewards, len(training_datums))

    ckpt = training_client.save_weights_for_sampler(name="rl-final").result()
    print(f"Saved: {ckpt.path}")
    return ckpt.path
