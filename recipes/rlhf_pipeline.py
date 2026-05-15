#!/usr/bin/env python3
"""MinT Recipe: RLHF Pipeline

A minimal real 3-stage RLHF pipeline:
  1. SFT with recipe.supervised.train.main()
  2. PRM/preference model with forward_backward_custom()
  3. RL with recipe.rl.train.main() and a MessageEnv that scores responses with
     the PRM SamplingClient when available, or a rule reward fallback.

Run:
  MINT_API_KEY=sk-xxx python recipes/rlhf_pipeline.py
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
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mint import types  # noqa: E402


MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
SFT_STEPS = int(os.environ.get("MINT_RLHF_SFT_STEPS", "1"))
PRM_STEPS = int(os.environ.get("MINT_RLHF_PRM_STEPS", "1"))
RL_STEPS = int(os.environ.get("MINT_RLHF_RL_STEPS", "1"))
BATCH_SIZE = int(os.environ.get("MINT_RLHF_BATCH", "2"))
GROUP_SIZE = int(os.environ.get("MINT_RLHF_GROUP", "2"))
MAX_TOKENS = int(os.environ.get("MINT_RLHF_MAX_TOKENS", "32"))
LOG_ROOT = Path(os.environ.get("MINT_LOG_ROOT", "/tmp"))


SFT_CONVERSATIONS = [
    [
        {"role": "user", "content": "Give one tip for reliable backups."},
        {"role": "assistant", "content": "Schedule automatic backups and test restores regularly."},
    ],
    [
        {"role": "user", "content": "What makes a code review useful?"},
        {"role": "assistant", "content": "A useful review is specific, actionable, and tied to a real risk."},
    ],
]

PREFERENCE_PAIRS = [
    (
        "Give one tip for reliable backups.",
        "Schedule automatic backups and test restores regularly.",
        "Backups are good.",
    ),
    (
        "What makes a code review useful?",
        "A useful review is specific, actionable, and tied to a real risk.",
        "It says the code is bad.",
    ),
]

RL_PROMPTS = [
    "Give one tip for reliable backups.",
    "What makes a code review useful?",
]


class ListSFTDataset(recipe.supervised.types.SupervisedDataset):
    def __init__(self, conversations, model_name, renderer_name, batch_size, max_length=512):
        tokenizer = get_tokenizer(model_name)
        renderer = recipe.renderers.get_renderer(renderer_name, tokenizer)
        self.datums = [
            recipe.supervised.conversation_to_datum(conv, renderer, max_length=max_length)
            for conv in conversations
        ]
        self.batch_size = batch_size

    def __len__(self):
        return max(1, (len(self.datums) + self.batch_size - 1) // self.batch_size)

    def get_batch(self, index):
        start = (index * self.batch_size) % len(self.datums)
        batch = self.datums[start : start + self.batch_size]
        if len(batch) < self.batch_size:
            batch += self.datums[: self.batch_size - len(batch)]
        return batch


@chz.chz
class ListSFTDatasetBuilder(recipe.supervised.types.SupervisedDatasetBuilder):
    conversations: list[list[dict[str, str]]]
    model_name: str
    renderer_name: str
    batch_size: int = BATCH_SIZE

    def __call__(self):
        return ListSFTDataset(self.conversations, self.model_name, self.renderer_name, self.batch_size), None


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
            tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        )
    return tokenizer.encode(f"User: {prompt}\nAssistant:", add_special_tokens=True)


def build_datum(prompt_tokens: list[int], completion_text: str, tokenizer: Any) -> types.Datum:
    completion_tokens = tokenizer.encode(f" {completion_text}", add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)
    all_tokens = prompt_tokens + completion_tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens),
        },
    )


def _to_float_tensor(value: Any) -> torch.Tensor:
    if hasattr(value, "to_torch"):
        return value.to_torch().flatten().float()
    if hasattr(value, "tolist"):
        return torch.tensor(value.tolist(), dtype=torch.float32).flatten()
    return torch.tensor(value, dtype=torch.float32).flatten()


def sequence_logprob(logprobs: Any, weights: Any) -> torch.Tensor:
    if isinstance(logprobs, torch.Tensor):
        logprob_tensor = logprobs.flatten().float()
    elif hasattr(logprobs, "to_torch"):
        logprob_tensor = logprobs.to_torch().flatten().float()
    else:
        logprob_tensor = torch.as_tensor(logprobs, dtype=torch.float32).flatten()
    return torch.dot(logprob_tensor, _to_float_tensor(weights))


def pairwise_preference_loss(data: list[types.Datum], logprobs_list: list[Any]):
    chosen_scores, rejected_scores = [], []
    for chosen_datum, rejected_datum, chosen_logprobs, rejected_logprobs in zip(
        data[::2], data[1::2], logprobs_list[::2], logprobs_list[1::2]
    ):
        chosen_scores.append(sequence_logprob(chosen_logprobs, chosen_datum.loss_fn_inputs["weights"]))
        rejected_scores.append(sequence_logprob(rejected_logprobs, rejected_datum.loss_fn_inputs["weights"]))
    margins = torch.stack(chosen_scores) - torch.stack(rejected_scores)
    loss = -F.logsigmoid(margins).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pair_accuracy": float((margins > 0).float().mean().detach().cpu()),
        "mean_margin": float(margins.mean().detach().cpu()),
    }


def build_preference_data(tokenizer: Any) -> list[types.Datum]:
    data: list[types.Datum] = []
    for prompt, chosen, rejected in PREFERENCE_PAIRS:
        prompt_tokens = build_prompt_tokens(prompt, tokenizer)
        data.append(build_datum(prompt_tokens, chosen, tokenizer))
        data.append(build_datum(prompt_tokens, rejected, tokenizer))
    return data


def _read_last_state_checkpoint(log_path: Path) -> str | None:
    checkpoints_path = log_path / "checkpoints.jsonl"
    if not checkpoints_path.exists():
        return None
    state_path = None
    for line in checkpoints_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = payload.get("state_path")
        if isinstance(value, str):
            state_path = value
    return state_path


async def run_sft_stage(renderer_name: str) -> str:
    print("\n=== Stage 1: SFT ===")
    log_path = LOG_ROOT / f"mint-rlhf-sft-{int(time.time())}"
    config = recipe.supervised.train.Config(
        log_path=str(log_path),
        model_name=MODEL,
        renderer_name=renderer_name,
        dataset_builder=ListSFTDatasetBuilder(
            conversations=SFT_CONVERSATIONS,
            model_name=MODEL,
            renderer_name=renderer_name,
            batch_size=BATCH_SIZE,
        ),
        learning_rate=1e-5,
        lora_rank=RANK,
        max_steps=SFT_STEPS,
        save_every=999,
        eval_every=999,
        infrequent_eval_every=999,
        ttl_seconds=3600,
    )
    await recipe.supervised.train.main(config=config)
    checkpoint_path = _read_last_state_checkpoint(log_path)
    if checkpoint_path is None:
        raise RuntimeError(f"SFT stage did not write a resumable checkpoint in {log_path}")
    print(f"SFT checkpoint for Stage 3: {checkpoint_path}")
    return checkpoint_path


def run_prm_stage(service_client: Any) -> Any | None:
    print("\n=== Stage 2: PRM / Preference Model ===")
    try:
        prm_client = service_client.create_lora_training_client(
            base_model=MODEL,
            rank=RANK,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        tokenizer = prm_client.get_tokenizer()
        data = build_preference_data(tokenizer)
        for step in range(1, PRM_STEPS + 1):
            result = prm_client.forward_backward_custom(data, pairwise_preference_loss).result()
            metrics = result.metrics or {}
            prm_client.optim_step(types.AdamParams(learning_rate=1e-5)).result()
            print(
                f"PRM step {step}: loss={metrics.get('loss', float('nan')):.6f}, "
                f"pair_accuracy={metrics.get('pair_accuracy', float('nan')):.2f}"
            )
        sampler = prm_client.save_weights_and_get_sampling_client(name="rlhf-prm")
        print("PRM sampler created from saved weights.")
        return sampler
    except Exception as exc:
        print(f"Warning: PRM stage failed; using rule-based reward fallback. Cause: {exc}")
        return None


def _score_with_prm(prm_sampler: Any, prompt: str, response: str) -> float:
    tokenizer = prm_sampler.get_tokenizer()
    scoring_prompt = f"Prompt: {prompt}\nResponse: {response}\nScore helpfulness briefly:"
    tokens = tokenizer.encode(scoring_prompt, add_special_tokens=True)
    result = prm_sampler.sample(
        prompt=types.ModelInput.from_ints(tokens=tokens),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=16, temperature=0.0, stop=[tokenizer.eos_token_id]),
    ).result()
    text = tokenizer.decode(result.sequences[0].tokens)
    if re.search(r"good|helpful|specific|backup|review|risk", text, re.I):
        return 0.5
    return 0.0


def rule_based_reward(response: str) -> float:
    return 0.5 if len(response.split()) >= 6 and any(
        word in response.lower() for word in ["specific", "automatic", "risk", "restore", "actionable"]
    ) else -0.1


class RLHFMessageEnv(MessageEnv):
    def __init__(self, prompt: str, prm_sampler: Any | None):
        self.prompt = prompt
        self.prm_sampler = prm_sampler

    async def initial_observation(self) -> list[dict]:
        return [
            {"role": "system", "content": "Answer with one specific, practical sentence."},
            {"role": "user", "content": self.prompt},
        ]

    async def step(self, message: dict) -> MessageStepResult:
        response = extract_content(message).strip()
        if self.prm_sampler is not None:
            reward = _score_with_prm(self.prm_sampler, self.prompt, response)
            reward_source = "prm"
        else:
            reward = rule_based_reward(response)
            reward_source = "rule"
        return MessageStepResult(
            reward=reward,
            episode_done=True,
            next_messages=[],
            metrics={"reward_source_prm": float(reward_source == "prm")},
        )


@dataclass(frozen=True)
class RLHFEnvGroupBuilder(recipe.rl.types.EnvGroupBuilder):
    prompt: str
    group_size: int
    renderer_name: str
    model_name: str
    prm_sampler: Any | None = None

    async def make_envs(self) -> Sequence[recipe.rl.types.Env]:
        tokenizer = get_tokenizer(self.model_name)
        renderer = recipe.renderers.get_renderer(self.renderer_name, tokenizer)
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=RLHFMessageEnv(self.prompt, self.prm_sampler),
                max_trajectory_tokens=512,
                max_generation_tokens=MAX_TOKENS,
            )
            for _ in range(self.group_size)
        ]

    def logging_tags(self) -> list[str]:
        return ["rlhf"]


class RLHFDataset(recipe.rl.types.RLDataset):
    def __init__(self, prompts, batch_size, group_size, renderer_name, model_name, prm_sampler):
        self.prompts = prompts
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.model_name = model_name
        self.prm_sampler = prm_sampler

    def __len__(self):
        return max(1, (len(self.prompts) + self.batch_size - 1) // self.batch_size)

    def get_batch(self, index):
        start = (index * self.batch_size) % len(self.prompts)
        batch = self.prompts[start : start + self.batch_size]
        if len(batch) < self.batch_size:
            batch += self.prompts[: self.batch_size - len(batch)]
        return [
            RLHFEnvGroupBuilder(
                prompt=prompt,
                group_size=self.group_size,
                renderer_name=self.renderer_name,
                model_name=self.model_name,
                prm_sampler=self.prm_sampler,
            )
            for prompt in batch
        ]


@chz.chz
class RLHFDatasetBuilder(recipe.rl.types.RLDatasetBuilder):
    prompts: list[str]
    batch_size: int
    group_size: int
    renderer_name: str
    model_name: str
    prm_sampler: Any | None = None

    async def __call__(self):
        return (
            RLHFDataset(
                self.prompts,
                self.batch_size,
                self.group_size,
                self.renderer_name,
                self.model_name,
                self.prm_sampler,
            ),
            None,
        )


async def run_rl_stage(renderer_name: str, prm_sampler: Any | None, sft_checkpoint_path: str) -> None:
    print("\n=== Stage 3: RL ===")
    config = recipe.rl.train.Config(
        learning_rate=1e-5,
        dataset_builder=RLHFDatasetBuilder(
            prompts=RL_PROMPTS,
            batch_size=1,
            group_size=GROUP_SIZE,
            renderer_name=renderer_name,
            model_name=MODEL,
            prm_sampler=prm_sampler,
        ),
        model_name=MODEL,
        load_checkpoint_path=sft_checkpoint_path,
        renderer_name=renderer_name,
        lora_rank=RANK,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        kl_penalty_coef=0.0,
        loss_fn="importance_sampling",
        log_path=str(LOG_ROOT / f"mint-rlhf-rl-{int(time.time())}"),
        max_steps=RL_STEPS,
        save_every=999,
        eval_every=999,
        ttl_seconds=3600,
        num_groups_to_log=1,
        rollout_json_export=False,
    )
    await recipe.rl.train.main(config=config)


def main() -> int:
    try:
        print("Connecting to MinT server...")
        print(f"Endpoint: {configured_base_url()}")
        service_client, capabilities = make_service_client()
        supported = supported_model_count(capabilities)
        print(f"Auth preflight: OK ({supported} supported models)" if supported is not None else "Auth preflight: OK")
        renderer_name = recipe.get_recommended_renderer_name(MODEL)
        print(f"Model: {MODEL}; renderer: {renderer_name}")

        sft_checkpoint_path = asyncio.run(run_sft_stage(renderer_name))
        prm_sampler = run_prm_stage(service_client)
        asyncio.run(run_rl_stage(renderer_name, prm_sampler, sft_checkpoint_path))
        print("\nRLHF pipeline complete.")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
