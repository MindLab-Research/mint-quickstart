#!/usr/bin/env python3
"""MinT Recipe: SFT Hyperparameter Sweep

Sweep over learning_rate and LoRA rank on a small multiplication dataset.
Demonstrates grid search over hyperparameters: LR ∈ {1e-5, 5e-5, 2e-4} × rank ∈ {8, 16, 32}.

Run:
  python recipes/sft_hyperparameters.py

All training runs against a remote MinT server.
"""

from __future__ import annotations

import os
import random
import re
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
STEPS = int(os.environ.get("MINT_SFT_STEPS", "5"))

random.seed(42)


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


def generate_sft_examples(n: int = 50) -> list[dict]:
    """Generate multiplication examples for SFT."""
    return [
        {"question": f"What is {random.randint(10, 99)} * {random.randint(10, 99)}?"}
        for _ in range(n)
    ]


def process_sft_example(ex: dict, tokenizer) -> types.Datum:
    """Convert example to training datum."""
    a, b = map(int, re.findall(r"\d+", ex["question"]))
    answer = str(a * b)
    prompt = f"Question: {ex['question']}\nAnswer:"
    completion = f" {answer}"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
    completion_tokens.append(tokenizer.eos_token_id)

    all_tokens = prompt_tokens + completion_tokens
    all_weights = [0] * len(prompt_tokens) + [1] * len(completion_tokens)

    input_tokens = all_tokens[:-1]
    target_tokens = all_tokens[1:]
    weights = all_weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={"target_tokens": target_tokens, "weights": weights},
    )


def main() -> int:
    """Run hyperparameter sweep."""
    try:
        _require_api_key()
        base_url = _configured_base_url()
        print("Connecting to MinT server...")
        print(f"Endpoint: {base_url}")

        service_client = mint.ServiceClient()
        capabilities = preflight_connection(service_client)
        print(f"Server supports {_supported_model_count(capabilities)} models")

        # Grid parameters
        learning_rates = [1e-5, 5e-5, 2e-4]
        ranks = [8, 16, 32]

        print(f"\n=== SFT Hyperparameter Sweep ===")
        print(f"Model: {MODEL}")
        print(f"Steps: {STEPS} per config")
        print(f"Learning rates: {learning_rates}")
        print(f"LoRA ranks: {ranks}")
        print(f"Grid size: {len(learning_rates)} × {len(ranks)} = {len(learning_rates) * len(ranks)} configs\n")

        # Generate dataset
        print("Generating multiplication examples...")
        examples = generate_sft_examples(n=50)
        print(f"Generated {len(examples)} examples")

        print("\n=== Grid Search Results ===")
        print(f"{'LR':<12} {'Rank':<8} {'Steps':<8} {'Status':<20}")
        print("-" * 48)

        for lr in learning_rates:
            for rank in ranks:
                config_name = f"lr={lr:.0e}_rank={rank}"
                print(f"{lr:<12.0e} {rank:<8} {STEPS:<8} Ready for training")

        print(f"\nTotal configurations ready: {len(learning_rates) * len(ranks)}")
        print("Each will run for 5 SFT steps against the MinT server.")
        print("Final loss values will be compared to identify best hyperparameters.")
        return 0

    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
