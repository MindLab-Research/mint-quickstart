#!/usr/bin/env python3
"""MinT Recipe: Multi-Agent RL

Create two independent LoRA training clients, let two agents sample responses in
a simple debate interaction, score both agents, and train both clients with
low-level TrainingClient APIs. If concurrent clients fail, fall back to one
role-switching client.

Run:
  MINT_API_KEY=sk-xxx python recipes/multi_agent_rl.py
"""

from __future__ import annotations

import os
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
RL_STEPS = int(os.environ.get("MINT_MULTI_AGENT_STEPS", "1"))
LR = float(os.environ.get("MINT_MULTI_AGENT_LR", "1e-5"))
MAX_TOKENS = int(os.environ.get("MINT_MULTI_AGENT_MAX_TOKENS", "32"))
TEMPERATURE = float(os.environ.get("MINT_MULTI_AGENT_TEMPERATURE", "0.7"))

TASKS = [
    {"question": "What is 2 + 3?", "answer": "5"},
    {"question": "What is 4 + 6?", "answer": "10"},
]


def _extract_number(text: str) -> str | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


def build_sft_datum(tokenizer: Any, prompt: str, target: str) -> types.Datum:
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f" {target}", add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)
    all_tokens = prompt_tokens + completion_tokens
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=all_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": all_tokens[1:],
            "weights": [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens),
        },
    )


def sample_text(sampling_client: Any, tokenizer: Any, prompt: str) -> str:
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


def evaluate_response(response: str, answer: str) -> float:
    return 1.0 if _extract_number(response) == answer else -0.25


def train_agent(training_client: Any, tokenizer: Any, prompt: str, answer: str, label: str) -> float:
    datum = build_sft_datum(tokenizer, prompt, answer)
    fb = training_client.forward_backward([datum], loss_fn="cross_entropy").result()
    training_client.optim_step(types.AdamParams(learning_rate=LR)).result()
    loss = 0.0
    total_weight = 0.0
    output = fb.loss_fn_outputs[0]
    logprobs = output["logprobs"]
    if hasattr(logprobs, "tolist"):
        logprobs = logprobs.tolist()
    weights = datum.loss_fn_inputs["weights"]
    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    for logprob, weight in zip(logprobs, weights):
        loss += -float(logprob) * float(weight)
        total_weight += float(weight)
    value = loss / max(total_weight, 1.0)
    print(f"  {label} train_cross_entropy={value:.6f}")
    return value


def create_two_agents(service_client: Any):
    agent_a = service_client.create_lora_training_client(
        base_model=MODEL,
        rank=RANK,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    agent_b = service_client.create_lora_training_client(
        base_model=MODEL,
        rank=RANK,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
    )
    return agent_a, agent_b


def run_two_agent_training(service_client: Any) -> None:
    print("\n=== Concurrent two-agent training ===")
    agent_a, agent_b = create_two_agents(service_client)
    tokenizer_a = agent_a.get_tokenizer()
    tokenizer_b = agent_b.get_tokenizer()
    print("Created two independent LoRA training clients.")

    for step in range(1, RL_STEPS + 1):
        task = TASKS[(step - 1) % len(TASKS)]
        question = task["question"]
        answer = task["answer"]
        print(f"\nStep {step}: {question} (answer={answer})")

        sampler_a = agent_a.save_weights_and_get_sampling_client(name=f"agent-a-step-{step}")
        sampler_b = agent_b.save_weights_and_get_sampling_client(name=f"agent-b-step-{step}")

        prompt_a = f"Agent A: answer the math question with only the number. {question}\nAnswer:"
        response_a = sample_text(sampler_a, tokenizer_a, prompt_a)
        reward_a = evaluate_response(response_a, answer)
        print(f"  Agent A response: {response_a[:120]} | reward={reward_a}")

        prompt_b = (
            f"Agent B: Agent A answered '{response_a}'. "
            f"Now answer the same question with only the number. {question}\nAnswer:"
        )
        response_b = sample_text(sampler_b, tokenizer_b, prompt_b)
        reward_b = evaluate_response(response_b, answer)
        print(f"  Agent B response: {response_b[:120]} | reward={reward_b}")

        # Low-level manual training: reinforce the known correct answer for both roles.
        train_agent(agent_a, tokenizer_a, prompt_a, answer, "Agent A")
        train_agent(agent_b, tokenizer_b, prompt_b, answer, "Agent B")

    ckpt_a = agent_a.save_weights_for_sampler(name="multi-agent-a-final").result()
    ckpt_b = agent_b.save_weights_for_sampler(name="multi-agent-b-final").result()
    print("\n=== Multi-Agent Summary ===")
    print("Mode: concurrent two-client")
    print(f"Agent A checkpoint: {ckpt_a.path}")
    print(f"Agent B checkpoint: {ckpt_b.path}")


def run_role_switching_fallback(service_client: Any, reason: Exception) -> None:
    print("\nWarning: concurrent LoRA path failed; falling back to single-agent role switching.")
    print(f"Reason: {reason}")
    agent = service_client.create_lora_training_client(base_model=MODEL, rank=RANK)
    tokenizer = agent.get_tokenizer()
    for step in range(1, RL_STEPS + 1):
        task = TASKS[(step - 1) % len(TASKS)]
        for role in ["Agent A", "Agent B"]:
            prompt = f"{role}: answer with only the number. {task['question']}\nAnswer:"
            train_agent(agent, tokenizer, prompt, task["answer"], role)
    ckpt = agent.save_weights_for_sampler(name="multi-agent-role-switch-final").result()
    print("\n=== Multi-Agent Summary ===")
    print("Mode: single-client role-switching fallback")
    print(f"Checkpoint: {ckpt.path}")


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
        print(f"Model: {MODEL}; rank: {RANK}; steps: {RL_STEPS}")

        try:
            run_two_agent_training(service_client)
        except Exception as exc:
            run_role_switching_fallback(service_client, exc)
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
