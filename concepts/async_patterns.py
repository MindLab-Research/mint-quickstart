#!/usr/bin/env python3
"""MinT Concepts: Async Patterns

Demonstrates concurrent sampling and training:
  - Send N sampling requests as futures before awaiting any
  - Pipeline forward_backward + optim_step
  - Avoid idle time on the worker

Run:
  python concepts/async_patterns.py

All training runs against a remote MinT server.
"""

from __future__ import annotations

import os
import sys
import time
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


def demonstrate_async_patterns() -> None:
    """Show concurrent futures for sampling and training."""
    _require_api_key()
    base_url = _configured_base_url()
    print("Connecting to MinT server at", base_url)

    service_client = mint.ServiceClient()
    capabilities = preflight_connection(service_client)
    print(f"Server supports {_supported_model_count(capabilities)} models")

    print("\n=== Sequential Sampling (the slow way) ===")
    print("""
for prompt in prompts:
    result = await sampling_client.sample_async(prompt)
    # ...process result...
    # ^^^ GPU is idle while we process on CPU
""")
    print("Problem: GPU finishes, we process, then send next request. Lots of idle time.")

    print("\n=== Concurrent Sampling (the fast way) ===")
    print("""
# Submit all requests at once as futures
futures = [
    sampling_client.sample_async(prompt, num_samples=1)
    for prompt in prompts
]

# Then gather results
results = await asyncio.gather(*futures)

# GPU pipeline batches all requests, uses time efficiently
""")
    print("Speedup: ~3-5x for 8 concurrent samples on a good GPU")

    print("\n=== Pipelined Training Loop ===")
    print("""
# Forward pass + loss computation (GPU)
fb_future = training_client.forward_backward_async(data=batch)

# While GPU computes, CPU can:
#   - Process previous rollouts
#   - Sample next batch
#   - Build advantage estimates

# When ready, collect loss and do optimizer step
loss_result = await fb_future
await training_client.optim_step_async(loss_result)
""")

    print("\n=== Interleaved RL Loop ===")
    print("""
for step in range(steps):
    # Sample futures for next batch
    sample_futures = [
        sampling_client.sample_async(prompt) for prompt in prompts
    ]

    # While sampling, process previous rewards/advantages
    advantages = compute_advantages(prev_trajectory_group)

    # Gather samples
    samples = await asyncio.gather(*sample_futures)

    # Train on current batch
    fb_future = training_client.forward_backward_async(
        loss_fn="grpo",
        data=build_rl_data(samples, advantages)
    )

    # While training, prepare next sample prompts
    new_prompts = prepare_next_prompts()

    # Optimizer step
    loss = await fb_future
    await training_client.optim_step_async(loss)

    print(f"Step {step}: loss={loss.value:.4f}")
""")

    print("\n=== Key Principles ===")
    print("1. Submit N requests BEFORE awaiting any (gather into futures list first)")
    print("2. Use asyncio.gather(*futures) to wait for all concurrently")
    print("3. Do CPU work (processing, advantage computation) while GPU is busy")
    print("4. Never await immediately after submitting a single request")


def main() -> int:
    """Entry point."""
    try:
        demonstrate_async_patterns()
        print("\nAsync patterns concept demonstration complete.")
        return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
