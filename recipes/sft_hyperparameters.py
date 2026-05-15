#!/usr/bin/env python3
"""MinT Recipe: SFT Hyperparameter Sweep

Run a small real supervised fine-tuning grid on MinT. The recipe sweeps
2 learning rates x 2 LoRA ranks, trains each config for 2 SFT steps by default,
and prints a summary of the final loss from each run.

Run:
  MINT_API_KEY=sk-xxx python recipes/sft_hyperparameters.py

Useful overrides:
  MINT_SFT_STEPS=1 MINT_SFT_LRS=1e-5,5e-5 MINT_LORA_RANKS=8,16 python recipes/sft_hyperparameters.py
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from _common import (  # noqa: E402
    configured_base_url,
    supported_model_count,
    make_service_client,
)

import chz  # noqa: E402
import mint.recipe as recipe  # noqa: E402
from mint.recipe import get_tokenizer  # noqa: E402


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
STEPS = int(os.environ.get("MINT_SFT_STEPS", "2"))
BATCH_SIZE = int(os.environ.get("MINT_SFT_BATCH", "4"))
MAX_LENGTH = int(os.environ.get("MINT_SFT_MAX_LENGTH", "512"))
DATASET_SIZE = int(os.environ.get("MINT_SFT_DATASET_SIZE", "16"))
LOG_ROOT = Path(os.environ.get("MINT_LOG_ROOT", "/tmp"))

random.seed(42)


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


def generate_sft_conversations(n: int) -> list[list[dict[str, str]]]:
    """Generate deterministic arithmetic chat conversations for SFT."""
    conversations: list[list[dict[str, str]]] = []
    for _ in range(n):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        conversations.append(
            [
                {"role": "user", "content": f"What is {a} * {b}?"},
                {"role": "assistant", "content": str(a * b)},
            ]
        )
    return conversations


class ArithmeticSFTDataset(recipe.supervised.types.SupervisedDataset):
    """Tiny in-memory supervised dataset using conversation_to_datum()."""

    def __init__(
        self,
        conversations: list[list[dict[str, str]]],
        model_name: str,
        renderer_name: str,
        batch_size: int,
        max_length: int,
    ):
        tokenizer = get_tokenizer(model_name)
        renderer = recipe.renderers.get_renderer(renderer_name, tokenizer)
        self.datums = [
            recipe.supervised.conversation_to_datum(
                conversation,
                renderer,
                max_length=max_length,
            )
            for conversation in conversations
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

    def set_epoch(self, seed: int = 0):
        rng = random.Random(seed)
        rng.shuffle(self.datums)


@chz.chz
class ArithmeticSFTDatasetBuilder(recipe.supervised.types.SupervisedDatasetBuilder):
    conversations: list[list[dict[str, str]]]
    model_name: str
    renderer_name: str
    batch_size: int = BATCH_SIZE
    max_length: int = MAX_LENGTH

    def __call__(self):
        return (
            ArithmeticSFTDataset(
                self.conversations,
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


async def run_config(
    *,
    conversations: list[list[dict[str, str]]],
    renderer_name: str,
    learning_rate: float,
    rank: int,
    run_id: str,
) -> dict[str, Any]:
    label = f"lr={learning_rate:.0e}, rank={rank}"
    safe_lr = f"{learning_rate:.0e}".replace("-", "m")
    log_path = LOG_ROOT / f"mint-sft-sweep-{run_id}-{safe_lr}-rank{rank}"

    print(f"\n--- Training config: {label} ---")
    print(f"Log path: {log_path}")

    config = recipe.supervised.train.Config(
        log_path=str(log_path),
        model_name=MODEL,
        renderer_name=renderer_name,
        dataset_builder=ArithmeticSFTDatasetBuilder(
            conversations=conversations,
            model_name=MODEL,
            renderer_name=renderer_name,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
        ),
        learning_rate=learning_rate,
        lora_rank=rank,
        max_steps=STEPS,
        save_every=999,
        eval_every=999,
        infrequent_eval_every=999,
        ttl_seconds=3600,
    )
    await recipe.supervised.train.main(config=config)
    final_loss = _read_final_train_loss(log_path)
    print(f"Completed {label}: final_train_mean_nll={final_loss}")
    return {
        "learning_rate": learning_rate,
        "rank": rank,
        "steps": STEPS,
        "log_path": str(log_path),
        "final_train_mean_nll": final_loss,
    }


async def run_sweep() -> list[dict[str, Any]]:
    learning_rates = _parse_float_list("MINT_SFT_LRS", [1e-5, 5e-5])
    ranks = _parse_int_list("MINT_LORA_RANKS", [8, 16])
    renderer_name = recipe.get_recommended_renderer_name(MODEL)
    conversations = generate_sft_conversations(DATASET_SIZE)
    run_id = str(int(time.time()))

    print("\n=== SFT Hyperparameter Sweep ===")
    print(f"Model:          {MODEL}")
    print(f"Renderer:       {renderer_name}")
    print(f"Steps/config:   {STEPS}")
    print(f"Batch size:     {BATCH_SIZE}")
    print(f"Dataset size:   {len(conversations)}")
    print(f"Learning rates: {learning_rates}")
    print(f"LoRA ranks:     {ranks}")
    print(f"Grid size:      {len(learning_rates)} x {len(ranks)} = {len(learning_rates) * len(ranks)} configs")

    results: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for rank in ranks:
            results.append(
                await run_config(
                    conversations=conversations,
                    renderer_name=renderer_name,
                    learning_rate=learning_rate,
                    rank=rank,
                    run_id=run_id,
                )
            )
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print("\n=== Grid Search Summary ===")
    print(f"{'LR':<12} {'Rank':<8} {'Steps':<8} {'Final train NLL':<18} {'Log path'}")
    print("-" * 90)
    for result in results:
        loss = result["final_train_mean_nll"]
        loss_text = f"{loss:.4f}" if isinstance(loss, float) else "n/a"
        print(
            f"{result['learning_rate']:<12.0e} "
            f"{result['rank']:<8} "
            f"{result['steps']:<8} "
            f"{loss_text:<18} "
            f"{result['log_path']}"
        )


def main() -> int:
    try:
        base_url = configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

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
