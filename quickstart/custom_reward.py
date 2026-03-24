#!/usr/bin/env python3
"""MinT custom reward quickstart.

Demonstrates the standard MinT RL pattern for custom rewards:
sample -> score in client Python -> compute advantages -> importance_sampling.

Run:
  python quickstart/custom_reward.py

All training runs against a remote MinT server.
This script does NOT start any backend services locally.
"""

from __future__ import annotations

import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


REPO_ROOT = Path(__file__).resolve().parents[1]
load_env_file(REPO_ROOT / ".env")

for base_dir in (REPO_ROOT.parent, REPO_ROOT):
    for src_dir in ("mindlab-toolkit-alpha/src", "mindlab-toolkit/src"):
        mint_src = base_dir / src_dir
        if mint_src.exists() and str(mint_src) not in sys.path:
            sys.path.insert(0, str(mint_src))
            break
    else:
        continue
    break

import mint
import tinker
from mint import types

MODEL = os.environ.get("MINT_BASE_MODEL", "Qwen/Qwen3-0.6B")
RANK = int(os.environ.get("MINT_LORA_RANK", "16"))
RL_STEPS = int(os.environ.get("MINT_CUSTOM_REWARD_STEPS", "8"))
RL_LR = float(os.environ.get("MINT_CUSTOM_REWARD_LR", "2e-5"))
RL_BATCH = int(os.environ.get("MINT_CUSTOM_REWARD_BATCH", "8"))
RL_GROUP = int(os.environ.get("MINT_CUSTOM_REWARD_GROUP", "6"))
MAX_TOK = int(os.environ.get("MINT_CUSTOM_REWARD_MAX_TOKENS", "16"))
TEMPERATURE = float(os.environ.get("MINT_CUSTOM_REWARD_TEMPERATURE", "0.8"))

random.seed(42)


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    format_reward: float
    distance_reward: float
    exact_bonus: float


def _configured_base_url() -> str:
    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        return "https://mint.macaron.xin/"
    return base_url


def _require_api_key() -> str:
    api_key = (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "MINT_API_KEY not found. Set `MINT_API_KEY=sk-your-api-key-here` in the shell "
        f"or add it to `{REPO_ROOT / '.env'}` before running custom_reward.py."
    )


def _status_code_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def preflight_connection(service_client: mint.ServiceClient):
    base_url = _configured_base_url()
    try:
        return service_client.get_server_capabilities()
    except tinker.APITimeoutError as exc:
        raise RuntimeError(
            f"Auth preflight timed out while contacting {base_url}. "
            "Check `MINT_BASE_URL` and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            f"Auth preflight could not reach {base_url}. "
            "Check `MINT_BASE_URL`, network access, and server status."
        ) from exc
    except tinker.APIStatusError as exc:
        status_code = _status_code_from_error(exc)
        if status_code in {401, 403}:
            raise RuntimeError(
                "Auth preflight was rejected by the MinT server "
                f"(HTTP {status_code}). Check that `MINT_API_KEY` is valid for {base_url}."
            ) from exc
        raise RuntimeError(
            "Auth preflight failed with an unexpected MinT server response "
            f"(HTTP {status_code or 'unknown'}) from {base_url}."
        ) from exc


def extract_prediction(response: str) -> int | None:
    match = re.search(r"-?\d+", response)
    return int(match.group()) if match else None


def generate_problem() -> tuple[str, int]:
    a = random.randint(10, 199)
    b = random.randint(10, 199)
    return f"What is {a} * {b}?", a * b


def compute_reward_breakdown(response: str, correct_answer: int) -> RewardBreakdown:
    prediction = extract_prediction(response)
    if prediction is None:
        return RewardBreakdown(total=0.0, format_reward=0.0, distance_reward=0.0, exact_bonus=0.0)

    format_reward = 0.2
    error = abs(prediction - correct_answer)
    distance_scale = max(abs(correct_answer), 20)
    closeness = max(0.0, 1.0 - min(error / distance_scale, 1.0))
    distance_reward = 0.5 * closeness
    exact_bonus = 0.3 if prediction == correct_answer else 0.0
    total = min(1.0, format_reward + distance_reward + exact_bonus)
    return RewardBreakdown(
        total=total,
        format_reward=format_reward,
        distance_reward=distance_reward,
        exact_bonus=exact_bonus,
    )


def summarize_breakdowns(breakdowns: list[RewardBreakdown]) -> dict[str, float]:
    if not breakdowns:
        return {
            "avg_reward": 0.0,
            "exact_rate": 0.0,
            "format_rate": 0.0,
            "avg_distance_reward": 0.0,
        }

    count = len(breakdowns)
    return {
        "avg_reward": sum(item.total for item in breakdowns) / count,
        "exact_rate": sum(1 for item in breakdowns if item.exact_bonus > 0) / count,
        "format_rate": sum(1 for item in breakdowns if item.format_reward > 0) / count,
        "avg_distance_reward": sum(item.distance_reward for item in breakdowns) / count,
    }


def build_rl_datum(
    prompt_tokens: list[int],
    response_tokens: list[int],
    logprobs: list[float],
    advantage: float,
) -> types.Datum:
    prefix_len = len(prompt_tokens) - 1
    full_tokens = prompt_tokens + response_tokens
    response_len = len(response_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": full_tokens[1:],
            "weights": [0.0] * prefix_len + [1.0] * response_len,
            "logprobs": [0.0] * prefix_len + logprobs,
            "advantages": [0.0] * prefix_len + [advantage] * response_len,
        },
    )


def main() -> int:
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        preflight_connection(service_client)

        training_client = service_client.create_lora_training_client(
            base_model=MODEL,
            rank=RANK,
            train_mlp=True,
            train_attn=True,
            train_unembed=True,
        )
        tokenizer = training_client.get_tokenizer()
        print(f"Model: {MODEL}, Vocab: {tokenizer.vocab_size:,}\n")

        for step in range(1, RL_STEPS + 1):
            sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"custom-reward-step-{step}"
            )

            training_datums: list[types.Datum] = []
            all_breakdowns: list[RewardBreakdown] = []

            for _ in range(RL_BATCH):
                question, answer = generate_problem()
                prompt_tokens = tokenizer.encode(f"Question: {question}\nAnswer:")

                result = sampling_client.sample(
                    prompt=types.ModelInput.from_ints(tokens=prompt_tokens),
                    num_samples=RL_GROUP,
                    sampling_params=types.SamplingParams(
                        max_tokens=MAX_TOK,
                        temperature=TEMPERATURE,
                        stop_token_ids=[tokenizer.eos_token_id],
                    ),
                ).result()

                group_rewards: list[float] = []
                group_sequences: list[list[int]] = []
                group_logprobs: list[list[float]] = []

                for seq in result.sequences:
                    response_tokens = list(seq.tokens)
                    response_text = tokenizer.decode(response_tokens)
                    breakdown = compute_reward_breakdown(response_text, answer)
                    group_rewards.append(breakdown.total)
                    group_sequences.append(response_tokens)
                    group_logprobs.append(list(seq.logprobs or [0.0] * len(response_tokens)))
                    all_breakdowns.append(breakdown)

                mean_reward = sum(group_rewards) / len(group_rewards)
                advantages = [reward - mean_reward for reward in group_rewards]

                for response_tokens, logprobs, advantage in zip(
                    group_sequences, group_logprobs, advantages
                ):
                    if not response_tokens or abs(advantage) < 1e-12:
                        continue
                    training_datums.append(
                        build_rl_datum(prompt_tokens, response_tokens, logprobs, advantage)
                    )

            if training_datums:
                training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling"
                ).result()
                training_client.optim_step(types.AdamParams(learning_rate=RL_LR)).result()

            summary = summarize_breakdowns(all_breakdowns)
            print(
                f"Step {step}: avg_reward={summary['avg_reward']:.3f}, "
                f"exact_rate={summary['exact_rate']:.1%}, "
                f"format_rate={summary['format_rate']:.1%}, "
                f"datums={len(training_datums)}"
            )

        final_checkpoint = training_client.save_weights_for_sampler(
            name="custom-reward-final"
        ).result()
        print(f"\nSaved: {final_checkpoint.path}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
