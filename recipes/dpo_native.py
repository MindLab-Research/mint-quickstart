#!/usr/bin/env python3
"""MinT Recipe: DPO with Custom Loss

Direct Preference Optimization using forward_backward_custom with Bradley-Terry loss.
Demonstrates preference learning: chosen > rejected with KL penalty term (dpo_beta).

Run:
  python recipes/dpo_native.py

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
DPO_STEPS = int(os.environ.get("MINT_DPO_STEPS", "5"))
DPO_BETA = float(os.environ.get("MINT_DPO_BETA", "0.1"))
DPO_LR = float(os.environ.get("MINT_DPO_LR", "1e-5"))


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


def build_preference_pairs() -> list[dict]:
    """Build preference pairs for DPO training.

    Returns list of dicts with 'prompt', 'chosen', 'rejected' fields.
    """
    return [
        {
            "prompt": "What is 5+3?",
            "chosen": " 8",
            "rejected": " 10",
        },
        {
            "prompt": "What is 10*2?",
            "chosen": " 20",
            "rejected": " 5",
        },
        {
            "prompt": "What is 100/5?",
            "chosen": " 20",
            "rejected": " 15",
        },
    ]


def main() -> int:
    """Run DPO training."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        print(f"\n=== DPO Training Setup ===")
        print(f"Model: {MODEL}")
        print(f"LoRA Rank: {RANK}")
        print(f"DPO Steps: {DPO_STEPS}")
        print(f"DPO Beta: {DPO_BETA} (KL penalty strength)")
        print(f"Learning Rate: {DPO_LR}")

        # Build preference data
        pairs = build_preference_pairs()
        print(f"\nPreference pairs: {len(pairs)}")
        for i, pair in enumerate(pairs):
            print(f"  {i+1}. Prompt: '{pair['prompt']}'")
            print(f"     Chosen:   '{pair['chosen']}'")
            print(f"     Rejected: '{pair['rejected']}'")

        print(f"\n=== DPO Loss Computation ===")
        print("For each (prompt, chosen, rejected) triple:")
        print("  1. Compute forward pass for chosen response")
        print("     -> chosen_logprob = log p_policy(chosen | prompt)")
        print("  2. Compute forward pass for rejected response")
        print("     -> rejected_logprob = log p_policy(rejected | prompt)")
        print("  3. Compute forward pass with reference model to get KL penalty")
        print("     -> ref_chosen_logprob = log p_ref(chosen | prompt)")
        print("     -> ref_rejected_logprob = log p_ref(rejected | prompt)")
        print("  4. DPO loss:")
        print(f"     L = -log sigmoid({DPO_BETA} * (chosen_logprob - rejected_logprob")
        print(f"                            - (ref_chosen_logprob - ref_rejected_logprob)))")
        print("     Higher margin -> lower loss")

        print(f"\nWill run {DPO_STEPS} DPO steps, tracking chosen_logprob - rejected_logprob")
        print("convergence = margin should increase over steps (policy learns to prefer chosen)")

        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
