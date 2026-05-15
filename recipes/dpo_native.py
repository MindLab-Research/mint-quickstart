#!/usr/bin/env python3
"""MinT Recipe: DPO with Custom Loss

Train on preference pairs using TrainingClient.forward_backward_custom() and a
Bradley-Terry pairwise preference loss. Datums are ordered chosen/rejected:
even index = chosen, odd index = rejected.

Run:
  MINT_API_KEY=sk-xxx python recipes/dpo_native.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Any

from _common import (  # noqa: E402
    configured_base_url,
    supported_model_count,
    make_service_client,
)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mint import types  # noqa: E402


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
DPO_STEPS = int(os.environ.get("MINT_DPO_STEPS", "3"))
DPO_LR = float(os.environ.get("MINT_DPO_LR", "1e-5"))


@dataclass(frozen=True)
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


PREFERENCE_PAIRS = [
    PreferencePair(
        prompt="Explain why regular backups matter.",
        chosen="Backups protect data by creating copies that can be restored after mistakes, hardware failures, or ransomware.",
        rejected="Backups are good.",
    ),
    PreferencePair(
        prompt="What is TCP?",
        chosen="TCP is a reliable transport protocol that provides ordered delivery of data between applications.",
        rejected="TCP is a network thing.",
    ),
    PreferencePair(
        prompt="Why use caching?",
        chosen="Caching stores frequently accessed data in faster storage so later requests can avoid repeated expensive work.",
        rejected="Caching makes things faster.",
    ),
    PreferencePair(
        prompt="What should a code review comment optimize for?",
        chosen="A code review comment should be specific, actionable, and tied to a real correctness or maintainability risk.",
        rejected="It should say the code is bad.",
    ),
]


def _coerce_chat_template_tokens(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        tokenized = tokenized["input_ids"]
    elif hasattr(tokenized, "input_ids"):
        tokenized = getattr(tokenized, "input_ids")
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, tuple):
        tokenized = list(tokenized)
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(token) for token in tokenized]


def build_prompt_tokens(prompt: str, tokenizer: Any) -> list[int]:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return _coerce_chat_template_tokens(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return tokenizer.encode(f"User: {prompt}\nAssistant:", add_special_tokens=True)


def build_datum(prompt_tokens: list[int], completion_text: str, tokenizer: Any) -> types.Datum:
    """Build a Datum with prompt tokens masked out and completion tokens trained."""
    completion_tokens = tokenizer.encode(f" {completion_text}", add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)

    all_tokens = prompt_tokens + completion_tokens
    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def flatten_preference_pairs(pairs: list[PreferencePair], tokenizer: Any) -> list[types.Datum]:
    data: list[types.Datum] = []
    for pair in pairs:
        prompt_tokens = build_prompt_tokens(pair.prompt, tokenizer)
        data.append(build_datum(prompt_tokens, pair.chosen, tokenizer))
        data.append(build_datum(prompt_tokens, pair.rejected, tokenizer))
    return data


def _to_float_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_torch"):
        tensor = value.to_torch()
    elif hasattr(value, "tolist"):
        tensor = torch.tensor(value.tolist(), dtype=torch.float32)
    else:
        tensor = torch.tensor(value, dtype=torch.float32)
    return tensor.flatten().float()


def sequence_logprob(logprobs: Any, weights: Any) -> torch.Tensor:
    """Weighted sequence logprob while preserving gradients on logprobs."""
    if isinstance(logprobs, torch.Tensor):
        logprob_tensor = logprobs.flatten().float()
    elif hasattr(logprobs, "to_torch"):
        logprob_tensor = logprobs.to_torch().flatten().float()
    else:
        logprob_tensor = torch.as_tensor(logprobs, dtype=torch.float32).flatten()

    weight_tensor = _to_float_tensor(weights)
    if logprob_tensor.shape != weight_tensor.shape:
        raise ValueError(
            "logprobs and weights must have the same shape, "
            f"got {tuple(logprob_tensor.shape)} and {tuple(weight_tensor.shape)}"
        )
    return torch.dot(logprob_tensor, weight_tensor)


def pairwise_preference_loss(data: list[types.Datum], logprobs_list: list[Any]):
    """Bradley-Terry loss over chosen/rejected datum pairs."""
    if len(data) % 2 != 0:
        raise ValueError(
            "pairwise_preference_loss expects an even number of datums ordered as "
            "(chosen, rejected) pairs."
        )

    chosen_scores: list[torch.Tensor] = []
    rejected_scores: list[torch.Tensor] = []
    for chosen_datum, rejected_datum, chosen_logprobs, rejected_logprobs in zip(
        data[::2], data[1::2], logprobs_list[::2], logprobs_list[1::2]
    ):
        chosen_scores.append(
            sequence_logprob(chosen_logprobs, chosen_datum.loss_fn_inputs["weights"])
        )
        rejected_scores.append(
            sequence_logprob(rejected_logprobs, rejected_datum.loss_fn_inputs["weights"])
        )

    chosen_tensor = torch.stack(chosen_scores)
    rejected_tensor = torch.stack(rejected_scores)
    margins = chosen_tensor - rejected_tensor
    loss = -F.logsigmoid(margins).mean()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "pair_accuracy": float((margins > 0).float().mean().detach().cpu()),
        "mean_margin": float(margins.mean().detach().cpu()),
        "mean_chosen_score": float(chosen_tensor.mean().detach().cpu()),
        "mean_rejected_score": float(rejected_tensor.mean().detach().cpu()),
    }
    return loss, metrics


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

        print("\n=== DPO Training Setup ===")
        print(f"Model:       {MODEL}")
        print(f"LoRA Rank:   {RANK}")
        print(f"DPO Steps:   {DPO_STEPS}")
        print(f"Learning LR: {DPO_LR}")
        print(f"Pairs:       {len(PREFERENCE_PAIRS)}")

        training_client = service_client.create_lora_training_client(
            base_model=MODEL,
            rank=RANK,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        tokenizer = training_client.get_tokenizer()
        data = flatten_preference_pairs(PREFERENCE_PAIRS, tokenizer)
        print(f"Datums:      {len(data)} ({len(data) // 2} pairs; even=chosen, odd=rejected)")

        last_metrics: dict[str, float] = {}
        for step in range(1, DPO_STEPS + 1):
            result = training_client.forward_backward_custom(
                data,
                pairwise_preference_loss,
            ).result()
            metrics = result.metrics or {}
            training_client.optim_step(types.AdamParams(learning_rate=DPO_LR)).result()
            loss = float(metrics.get("loss", float("nan")))
            pair_accuracy = float(metrics.get("pair_accuracy", float("nan")))
            mean_margin = float(metrics.get("mean_margin", float("nan")))
            print(
                f"Step {step}: loss={loss:.6f}, "
                f"pair_accuracy={pair_accuracy:.2f}, mean_margin={mean_margin:.6f}"
            )
            last_metrics = {
                "loss": loss,
                "pair_accuracy": pair_accuracy,
                "mean_margin": mean_margin,
            }

        if not math.isfinite(last_metrics.get("loss", float("nan"))):
            raise RuntimeError(f"DPO loss was not finite: {last_metrics}")
        pair_accuracy = last_metrics.get("pair_accuracy", float("nan"))
        if not 0.0 <= pair_accuracy <= 1.0:
            raise RuntimeError(f"pair_accuracy out of range: {last_metrics}")

        final_checkpoint = training_client.save_weights_for_sampler(name="dpo-native-final").result()
        print("\n=== DPO Summary ===")
        print(f"Final loss:          {last_metrics['loss']:.6f}")
        print(f"Final pair_accuracy: {last_metrics['pair_accuracy']:.2f}")
        print(f"Final mean_margin:   {last_metrics['mean_margin']:.6f}")
        print(f"Sampler checkpoint:  {final_checkpoint.path}")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
