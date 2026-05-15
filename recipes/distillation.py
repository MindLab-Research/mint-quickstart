#!/usr/bin/env python3
"""MinT Recipe: Prompt Distillation

Sample answers from a teacher model, convert those answers into supervised
chat data, then SFT-train a smaller student model with recipe.supervised.

Run:
  MINT_API_KEY=sk-xxx python recipes/distillation.py

Useful overrides:
  MINT_TEACHER_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 MINT_SFT_STEPS=1 python recipes/distillation.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from _common import (  # noqa: E402
    configured_base_url,
    is_model_supported,
    supported_model_count,
    make_service_client,
)

import chz  # noqa: E402
import mint.recipe as recipe  # noqa: E402
from mint import types  # noqa: E402
from mint.recipe import get_tokenizer  # noqa: E402


STUDENT_MODEL = os.environ.get("MINT_STUDENT_MODEL") or os.environ.get(
    "MINT_BASE_MODEL", "Qwen/Qwen3-0.6B"
)
REQUESTED_TEACHER_MODEL = os.environ.get(
    "MINT_TEACHER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"
)
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "2"))
BATCH_SIZE = int(os.environ.get("MINT_DISTILL_BATCH", "4"))
MAX_LENGTH = int(os.environ.get("MINT_DISTILL_MAX_LENGTH", "768"))
MAX_TOKENS = int(os.environ.get("MINT_DISTILL_MAX_TOKENS", "64"))
TEMPERATURE = float(os.environ.get("MINT_DISTILL_TEMPERATURE", "0.2"))
PROMPT_LIMIT = int(os.environ.get("MINT_DISTILL_PROMPTS", "4"))
LOG_ROOT = Path(os.environ.get("MINT_LOG_ROOT", "/tmp"))


PROMPTS = [
    "Give one practical tip for keeping regular backups.",
    "Explain TCP in one sentence.",
    "Why can caching make an app faster?",
    "Give one reason code reviews should be specific.",
    "What is a simple way to check an API timeout?",
]


class DistilledSFTDataset(recipe.supervised.types.SupervisedDataset):
    """Supervised dataset built from teacher-generated responses."""

    def __init__(
        self,
        examples: list[dict[str, str]],
        model_name: str,
        renderer_name: str,
        batch_size: int,
        max_length: int,
    ):
        tokenizer = get_tokenizer(model_name)
        renderer = recipe.renderers.get_renderer(renderer_name, tokenizer)
        self.datums = [
            recipe.supervised.conversation_to_datum(
                [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["teacher_response"]},
                ],
                renderer,
                max_length=max_length,
            )
            for item in examples
        ]
        self.batch_size = batch_size

    def __len__(self) -> int:
        return max(1, (len(self.datums) + self.batch_size - 1) // self.batch_size)

    def get_batch(self, index: int):
        start = (index * self.batch_size) % len(self.datums)
        batch = self.datums[start : start + self.batch_size]
        if len(batch) < self.batch_size:
            batch += self.datums[: self.batch_size - len(batch)]
        return batch


@chz.chz
class DistilledSFTDatasetBuilder(recipe.supervised.types.SupervisedDatasetBuilder):
    examples: list[dict[str, str]]
    model_name: str
    renderer_name: str
    batch_size: int = BATCH_SIZE
    max_length: int = MAX_LENGTH

    def __call__(self):
        return (
            DistilledSFTDataset(
                self.examples,
                self.model_name,
                self.renderer_name,
                self.batch_size,
                self.max_length,
            ),
            None,
        )


def _read_final_train_loss(log_path: Path) -> float | None:
    metrics_path = log_path / "metrics.jsonl"
    if not metrics_path.exists():
        return None
    final_loss = None
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = payload.get("train_mean_nll")
        if isinstance(value, int | float):
            final_loss = float(value)
    return final_loss


def _select_teacher_model(capabilities: object) -> str:
    support = is_model_supported(capabilities, REQUESTED_TEACHER_MODEL)
    if support is not False:
        return REQUESTED_TEACHER_MODEL

    print(
        "Warning: requested teacher model is not listed by this MinT server: "
        f"{REQUESTED_TEACHER_MODEL}. Falling back to self-distillation with {STUDENT_MODEL}."
    )
    return STUDENT_MODEL


def _decode_sample_text(sequence: object, tokenizer: Any) -> str:
    tokens = getattr(sequence, "tokens", [])
    text = tokenizer.decode(tokens).strip()
    return text or "I do not know yet."


def sample_teacher_responses(service_client, teacher_model: str, prompts: list[str]) -> list[dict[str, str]]:
    """Sample one teacher response per prompt using a real SamplingClient."""
    print("\n=== Stage 1: Teacher Sampling ===")
    print(f"Teacher model: {teacher_model}")
    print(f"Prompts:       {len(prompts)}")

    sampling_client = service_client.create_sampling_client(base_model=teacher_model)
    tokenizer = sampling_client.get_tokenizer()
    examples: list[dict[str, str]] = []

    for index, prompt in enumerate(prompts, 1):
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        result = sampling_client.sample(
            prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stop=[tokenizer.eos_token_id],
            ),
        ).result()
        response = _decode_sample_text(result.sequences[0], tokenizer)
        examples.append({"prompt": prompt, "teacher_response": response})
        print(f"[{index}/{len(prompts)}] prompt: {prompt}")
        print(f"          teacher: {response[:160]}")

    return examples


async def train_student(examples: list[dict[str, str]]) -> dict[str, Any]:
    """Train the student model on teacher outputs with recipe.supervised."""
    print("\n=== Stage 2: Student SFT ===")
    renderer_name = recipe.get_recommended_renderer_name(STUDENT_MODEL)
    log_path = LOG_ROOT / f"mint-distillation-{int(time.time())}"
    print(f"Student model: {STUDENT_MODEL}")
    print(f"Renderer:      {renderer_name}")
    print(f"Examples:      {len(examples)}")
    print(f"Steps:         {SFT_STEPS}")
    print(f"Log path:      {log_path}")

    config = recipe.supervised.train.Config(
        log_path=str(log_path),
        model_name=STUDENT_MODEL,
        renderer_name=renderer_name,
        dataset_builder=DistilledSFTDatasetBuilder(
            examples=examples,
            model_name=STUDENT_MODEL,
            renderer_name=renderer_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        ),
        learning_rate=float(os.environ.get("MINT_DISTILL_LR", "1e-5")),
        lora_rank=RANK,
        max_steps=SFT_STEPS,
        save_every=999,
        eval_every=999,
        infrequent_eval_every=999,
        ttl_seconds=3600,
    )
    await recipe.supervised.train.main(config=config)
    final_loss = _read_final_train_loss(log_path)
    return {"log_path": str(log_path), "final_train_mean_nll": final_loss}


def main() -> int:
    try:
        base_url = configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client, capabilities = make_service_client()
        supported = supported_model_count(capabilities)
        if supported is None:
            print("Auth preflight: OK")
        else:
            print(f"Auth preflight: OK ({supported} supported models)")

        teacher_model = _select_teacher_model(capabilities)
        prompts = PROMPTS[:PROMPT_LIMIT]
        examples = sample_teacher_responses(service_client, teacher_model, prompts)
        result = asyncio.run(train_student(examples))

        print("\n=== Distillation Summary ===")
        print(f"Teacher model:          {teacher_model}")
        print(f"Student model:          {STUDENT_MODEL}")
        print(f"Teacher samples:        {len(examples)}")
        print(f"Student SFT steps:      {SFT_STEPS}")
        print(f"Final train mean NLL:   {result['final_train_mean_nll']}")
        print(f"Log path:               {result['log_path']}")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
