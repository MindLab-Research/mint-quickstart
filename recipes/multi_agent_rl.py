#!/usr/bin/env python3
"""MinT Recipe: Multi-Agent RL

Multi-agent training where two policies interact with each other or environment.
Demonstrates self-play and symmetric/asymmetric reward patterns.

Run:
  python recipes/multi_agent_rl.py

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
    """Run multi-agent RL training."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== Multi-Agent RL Setup ===")
        print(f"Model: {MODEL}")
        print(f"LoRA Rank: {RANK}")
        print(f"RL Steps: {RL_STEPS}")

        print("\n=== Self-Play Pattern ===")
        print("Two agents compete or cooperate:")
        print("  1. Create two sampling clients (both from same base model, different LoRA)")
        print("  2. Agent A generates response to prompt")
        print("  3. Agent B generates response to prompt + Agent A's response")
        print("  4. Evaluate outcome (who answered better, or did they cooperate)")
        print("  5. Assign rewards to both agents")
        print("  6. Train both in parallel")

        print("\n=== Debate Example ===")
        print("Prompt: 'What is 2+3?'")
        print("  Agent A response: ' 5'")
        print("  Agent B response: ' 5' (agrees)")
        print("  Judge outcome: Both correct")
        print("  Reward: +1.0 for both")
        print("")
        print("Alternative:")
        print("  Agent A response: ' 4'")
        print("  Agent B response: ' 5' (disagrees)")
        print("  Judge outcome: B is correct")
        print("  Reward: +0.0 for A, +1.0 for B")

        print("\n=== Training Two Agents ===")
        print("Both can share the same base model but have separate LoRA weights:")
        print("""
# Create separate LoRA training clients
lora_a = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK, name="agent_a"
)
lora_b = await service_client.create_lora_training_client_async(
    base_model=MODEL, rank=RANK, name="agent_b"
)

# Create separate sampling clients for each LoRA
sampler_a = await service_client.create_sampling_client_async(
    weights=lora_a_weights
)
sampler_b = await service_client.create_sampling_client_async(
    weights=lora_b_weights
)

# Training loop: alternate or parallel training
for step in range(RL_STEPS):
    # Sample both agents concurrently
    samples_a = await sampler_a.sample_async(prompt)
    samples_b = await sampler_b.sample_async(prompt + samples_a.text)

    # Compute rewards and advantages
    reward_a, reward_b = evaluate_interaction(samples_a, samples_b)

    # Train both in parallel
    await asyncio.gather(
        lora_a.forward_backward(...),
        lora_b.forward_backward(...)
    )
""")

        print("\n=== Asymmetric Rewards ===")
        print("Agents don't need symmetric rewards:")
        print("  Winner: +1.0")
        print("  Loser:  -0.5")
        print("  Tie:    +0.0 for both")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
