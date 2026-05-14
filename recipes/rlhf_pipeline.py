#!/usr/bin/env python3
"""MinT Recipe: RLHF Pipeline

3-stage RLHF training:
  1. SFT: supervised fine-tune on chat dataset
  2. PRM: train preference reward model using Bradley-Terry loss
  3. RL: GRPO with preference model as reward function

Run:
  python recipes/rlhf_pipeline.py

All training runs against a remote MinT server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


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
SFT_STEPS = int(os.environ.get("MINT_SFT_STEPS", "3"))
PRM_STEPS = int(os.environ.get("MINT_PRM_STEPS", "3"))
RL_STEPS = int(os.environ.get("MINT_RL_STEPS", "3"))


def _configured_base_url() -> str:
    base_url = os.environ.get("MINT_BASE_URL") or os.environ.get("TINKER_BASE_URL")
    if not base_url:
        base_url = "https://mint.macaron.xin/"
    return base_url


def _require_api_key() -> str:
    api_key = (os.environ.get("MINT_API_KEY") or os.environ.get("TINKER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raise RuntimeError(
        "MINT_API_KEY not found. Set `MINT_API_KEY=sk-your-api-key-here` in the shell "
        f"or add it to `{REPO_ROOT / '.env'}` before running this script."
    )


def _status_code_from_error(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _supported_model_count(capabilities: object) -> int | None:
    models = getattr(capabilities, "supported_models", None)
    return len(models) if isinstance(models, list) else None


def preflight_connection(service_client: mint.ServiceClient):
    base_url = _configured_base_url()
    try:
        return service_client.get_server_capabilities()
    except tinker.APITimeoutError as exc:
        raise RuntimeError(
            "Auth preflight timed out while contacting "
            f"{base_url}. Check `MINT_BASE_URL` and retry."
        ) from exc
    except tinker.APIConnectionError as exc:
        raise RuntimeError(
            "Auth preflight could not reach "
            f"{base_url}. Check `MINT_BASE_URL`, network access, and server status."
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


def main() -> int:
    """Run 3-stage RLHF pipeline."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== 3-Stage RLHF Pipeline ===")
        print(f"Model: {MODEL}")
        print(f"LoRA Rank: {RANK}")

        print(f"\n=== Stage 1: SFT (Supervised Fine-Tuning) ===")
        print(f"Steps: {SFT_STEPS}")
        print(f"""
# Train on high-quality chat dataset
sft_client = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK
)

for step in range({SFT_STEPS}):
    # Mini-batch of (prompt, response) pairs
    batch = load_sft_batch()
    loss = await sft_client.forward_backward(
        loss_fn="cross_entropy",
        data=batch
    )
    await sft_client.optim_step(loss)
    print(f"SFT Step {{step}}: loss={{loss.value:.4f}}")

sft_weights = await sft_client.save_weights_and_get_sampling_client()
""")

        print(f"\n=== Stage 2: Train Preference Reward Model ===")
        print(f"Steps: {PRM_STEPS}")
        print(f"""
# Train a separate model to predict preference: p(A > B)
# Uses Bradley-Terry loss on preference pairs
prm_client = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK
)

preference_pairs = load_preference_data()  # List of (prompt, chosen, rejected)

for step in range({PRM_STEPS}):
    batch = sample_preference_batch(preference_pairs)

    # Custom loss: Bradley-Terry preference loss
    loss = await prm_client.forward_backward_custom(
        loss_fn=bradley_terry_loss,
        data=batch
    )
    await prm_client.optim_step(loss)
    print(f"PRM Step {{step}}: loss={{loss.value:.4f}}")

prm_weights = await prm_client.save_weights_and_get_sampling_client()
""")

        print(f"\n=== Stage 3: RL with Preference Reward Model ===")
        print(f"Steps: {RL_STEPS}")
        print(f"""
# Use PRM as reward function for GRPO
rl_client = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK
)

# Create evaluator using PRM
def prm_reward_fn(prompt, response):
    '''Score response using preference model'''
    model_input = renderer.build_generation_prompt([
        {{"role": "user", "content": prompt}},
        {{"role": "assistant", "content": response}}
    ])
    logprobs = await prm_client.forward_pass(model_input)
    return logprobs[-1]  # Score of last token

for step in range({RL_STEPS}):
    # Sample from policy
    samples = await policy_sampler.sample_async(prompt, num_samples=4)

    # Score with PRM
    rewards = [prm_reward_fn(prompt, s.text) for s in samples]

    # Compute advantages
    advantages = compute_advantages(rewards)

    # Train on samples
    rl_data = build_rl_data(samples, advantages)
    loss = await rl_client.forward_backward(
        loss_fn="grpo",
        data=rl_data
    )
    await rl_client.optim_step(loss)
    print(f"RL Step {{step}}: loss={{loss.value:.4f}}, mean_reward={{np.mean(rewards):.4f}}")
""")

        print(f"\n=== Expected Outcomes ===")
        print(f"Stage 1 (SFT):  Loss decreases from ~2.0 to ~1.0")
        print(f"Stage 2 (PRM):  Loss decreases from ~0.7 to ~0.3 (preference accuracy)")
        print(f"Stage 3 (RL):   Mean reward increases, loss improves with exploration")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
