#!/usr/bin/env python3
"""MinT Recipe: LoRA Adapter Checkpoint

Train a small LoRA adapter, save it with real MinT checkpoint APIs, then create
a SamplingClient from the saved weights and sample from it.

Run:
  MINT_API_KEY=sk-xxx python recipes/lora_adapter.py
"""

from __future__ import annotations

import os
import random
import re
import sys
from typing import Any

from _common import (  # noqa: E402
    configured_base_url,
    supported_model_count,
    make_service_client,
)

from mint import types  # noqa: E402


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_LORA_SFT_STEPS", "1"))
SFT_LR = float(os.environ.get("MINT_LORA_SFT_LR", "5e-5"))
MAX_TOKENS = int(os.environ.get("MINT_LORA_SAMPLE_MAX_TOKENS", "32"))
TEMPERATURE = float(os.environ.get("MINT_LORA_SAMPLE_TEMPERATURE", "0.0"))

random.seed(42)


def generate_sft_examples(n: int = 8) -> list[dict[str, str]]:
    examples = []
    for _ in range(n):
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        examples.append({"question": f"What is {a} + {b}?", "answer": str(a + b)})
    return examples


def process_sft_example(example: dict[str, str], tokenizer: Any) -> types.Datum:
    prompt = f"Question: {example['question']}\nAnswer:"
    completion = f" {example['answer']}"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)

    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def compute_cross_entropy(fb_result: Any, datums: list[types.Datum]) -> float:
    total_loss = 0.0
    total_weight = 0.0
    for index, output in enumerate(fb_result.loss_fn_outputs):
        logprobs = output["logprobs"]
        if hasattr(logprobs, "tolist"):
            logprobs = logprobs.tolist()
        weights = datums[index].loss_fn_inputs["weights"]
        if hasattr(weights, "tolist"):
            weights = weights.tolist()
        for logprob, weight in zip(logprobs, weights):
            total_loss += -float(logprob) * float(weight)
            total_weight += float(weight)
    return total_loss / max(total_weight, 1.0)


def _sample_text(sampling_client: Any, tokenizer: Any, prompt: str) -> str:
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
    return tokenizer.decode(result.sequences[0].tokens).strip()


def main() -> int:
    try:
        print("Connecting to MinT server...")
        print(f"Endpoint: {configured_base_url()}")
        service_client, capabilities = make_service_client()
        supported = supported_model_count(capabilities)
        if supported is None:
            print("Auth preflight: OK")
        else:
            print(f"Auth preflight: OK ({supported} supported models)")

        print("\n=== LoRA Adapter Checkpoint ===")
        print(f"Base model: {MODEL}")
        print(f"LoRA rank:  {RANK}")
        print(f"SFT steps:  {SFT_STEPS}")

        training_client = service_client.create_lora_training_client(
            base_model=MODEL,
            rank=RANK,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        tokenizer = training_client.get_tokenizer()
        data = [process_sft_example(example, tokenizer) for example in generate_sft_examples()]
        print(f"Training examples: {len(data)}")

        for step in range(1, SFT_STEPS + 1):
            fb_result = training_client.forward_backward(data, loss_fn="cross_entropy").result()
            loss = compute_cross_entropy(fb_result, data)
            training_client.optim_step(types.AdamParams(learning_rate=SFT_LR)).result()
            print(f"Step {step}: train_cross_entropy={loss:.6f}")

        print("\nSaving training state with save_state()...")
        state_checkpoint = training_client.save_state(name="lora-adapter-state").result()
        print(f"State checkpoint: {state_checkpoint.path}")

        print("\nSaving sampler weights with save_weights_for_sampler()...")
        sampler_checkpoint = training_client.save_weights_for_sampler(
            name="lora-adapter-sampler"
        ).result()
        print(f"Sampler weights:  {sampler_checkpoint.path}")

        print("\nCreating SamplingClient from saved weights...")
        sampling_client = service_client.create_sampling_client(
            model_path=sampler_checkpoint.path,
            base_model=MODEL,
        )
        sample_prompt = "Question: What is 4 + 5?\nAnswer:"
        response = _sample_text(sampling_client, tokenizer, sample_prompt)
        print(f"Sample prompt:   {sample_prompt}")
        print(f"Sample response: {response}")

        print("\n=== LoRA Adapter Summary ===")
        print("Trained adapter: yes")
        print(f"State checkpoint: {state_checkpoint.path}")
        print(f"Sampler checkpoint: {sampler_checkpoint.path}")
        print("Sampling from saved weights: success")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
