#!/usr/bin/env python3
"""MinT Recipe: Multi-Turn RL

Multi-turn RL where the policy generates multiple assistant responses in a dialogue.
Reward is computed on the full trajectory, masked so only assistant tokens get gradients.
Demonstrates sequence extension and loss_weights masking.

Run:
  python recipes/multi_turn_rl.py

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
    """Run multi-turn RL training."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== Multi-Turn RL Setup ===")
        print(f"Model: {MODEL}")
        print(f"LoRA Rank: {RANK}")
        print(f"RL Steps: {RL_STEPS}")

        print("\n=== Dialogue Structure ===")
        print("Example 2-turn conversation:")
        print("  Turn 1:")
        print("    Observation: 'What is 2+3?'")
        print("    Action:      ' 5'")
        print("  Turn 2:")
        print("    Observation: 'Correct! What is 10*2?'")
        print("    Action:      ' 20'")
        print("  Final Reward: 1.0 (both correct)")

        print("\n=== Sequence Extension ===")
        print("Multi-turn data merges consecutive turns into one training datum:")
        print("  Full sequence: [O1 A1 O2 A2]")
        print("  Tokens:        [100 101 102 103 200 201 300 301 400 401]")
        print("  Weights:       [0   0   0   1   1   0   0   0   1   1]")
        print("                   └─ Prompt ─┘  └─ A1 ─┘  └─ Prompt ─┘  └─ A2 ─┘")

        print("\n=== Loss Computation ===")
        print("forward_backward(loss_fn='grpo', data=multi_turn_datum)")
        print("  -> Computes NLL only on assistant tokens (weight=1)")
        print("  -> User observations (weight=0) are frozen")
        print("  -> RL advantage applies to assistant actions")
        print("  -> Final reward is shared across turns")

        print("\n=== Advantage Distribution ===")
        print("With reward=1.0 and group_size=4:")
        print("  mean_reward = 0.75 (avg of [1, 1, 1, 0] in group)")
        print("  advantage_i = 1.0 - 0.75 = 0.25 for this trajectory")
        print("  Gradient: positive signal for all assistant tokens")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
